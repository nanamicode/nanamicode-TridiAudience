package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	discoveryPort = 8790
	uploadPort    = 8791
	discoverText  = "TRIDI_COLLECTOR_DISCOVER_V1"
	hereText      = "TRIDI_COLLECTOR_HERE|8791"
	maxImageBytes = 16 << 20
	maxTextBytes  = 1 << 20
)

var version = "dev"

type box struct{ X, Y, W, H float64 }

type event struct {
	Type    string
	UUID    string
	Number  int64
	Gender  string
	Date    time.Time
	DateRaw string
	Track   string
	FrameW  int
	FrameH  int
	BBox    box
	Session string
	RawLine string
}

type server struct {
	dataRoot string
	mu       sync.Mutex
}

func main() {
	data := flag.String("data", "DADOS_TVBOX", "pasta raiz dos dados")
	flag.Parse()

	abs, err := filepath.Abs(*data)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		log.Fatal(err)
	}

	s := &server{dataRoot: abs}

	fmt.Println()
	fmt.Println("====================================================")
	fmt.Printf(" TRIDI COLLECTOR SERVER %s\n", version)
	fmt.Println("====================================================")
	fmt.Printf("Dados:           %s\n", abs)
	fmt.Printf("Discovery UDP:   %d\n", discoveryPort)
	fmt.Printf("Uploads HTTP:    %d\n", uploadPort)
	fmt.Printf("IPs deste PC:    %s\n", strings.Join(localIPv4s(), ", "))
	fmt.Println()
	fmt.Println("Deixe esta janela aberta. A TV Box encontra o servidor automaticamente.")
	fmt.Println("ALCANCES_MONTADOS: retangulo corporal + data.")
	fmt.Println("IMPRESSOES_MONTADAS: retangulo facial por genero + data.")
	fmt.Println()

	go runDiscovery()

	mux := http.NewServeMux()
	mux.HandleFunc("/ping", s.handlePing)
	mux.HandleFunc("/image", s.handleImage)
	mux.HandleFunc("/log", s.handleLog)
	mux.HandleFunc("/inference", s.handleInference)
	mux.HandleFunc("/crop", s.handleCrop)
	mux.HandleFunc("/end", s.handleEnd)

	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", uploadPort),
		Handler:           mux,
		ReadHeaderTimeout: 4 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       20 * time.Second,
	}
	log.Fatal(srv.ListenAndServe())
}

func runDiscovery() {
	addr := &net.UDPAddr{IP: net.IPv4zero, Port: discoveryPort}
	c, err := net.ListenUDP("udp4", addr)
	if err != nil {
		log.Printf("[ERRO] discovery UDP: %v", err)
		return
	}
	defer c.Close()
	buf := make([]byte, 512)
	for {
		n, remote, err := c.ReadFromUDP(buf)
		if err != nil {
			log.Printf("[UDP] %v", err)
			continue
		}
		if strings.TrimSpace(string(buf[:n])) != discoverText {
			continue
		}
		_, _ = c.WriteToUDP([]byte(hereText), remote)
		log.Printf("[DISCOVERY] TV Box %s", remote.IP)
	}
}

func (s *server) handlePing(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, "TRIDI_COLLECTOR_OK")
}

func (s *server) handleImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "METHOD", http.StatusMethodNotAllowed)
		return
	}
	session := safeName(query(r.URL, "session", "sem_sessao"))
	name := safeName(query(r.URL, "name", fmt.Sprintf("frame_%d.jpg", time.Now().UnixMilli())))
	if !strings.HasSuffix(strings.ToLower(name), ".jpg") && !strings.HasSuffix(strings.ToLower(name), ".jpeg") {
		name += ".jpg"
	}
	body, err := readLimited(w, r, maxImageBytes)
	if err != nil {
		return
	}
	dir := s.sessionDir(session)
	if err := atomicWrite(filepath.Join(dir, name), body, 0o644); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	log.Printf("[IMG] %s -> %s", session, name)
	ok(w)
}

func (s *server) handleLog(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "METHOD", http.StatusMethodNotAllowed)
		return
	}
	session := safeName(query(r.URL, "session", "sem_sessao"))
	body, err := readLimited(w, r, maxTextBytes)
	if err != nil {
		return
	}
	line := strings.TrimSpace(string(body))
	dir := s.sessionDir(session)

	s.mu.Lock()
	defer s.mu.Unlock()

	_ = appendLine(filepath.Join(dir, "telemetry.txt"), time.Now().Format("2006-01-02 15:04:05.000")+" | "+line)

	ev, err := parseEvent(line, session)
	if err != nil {
		log.Printf("[LOG] %s (sem montagem: %v)", session, err)
		ok(w)
		return
	}
	if err := s.mountEvent(dir, ev); err != nil {
		// Montage is post-processing. Never make Android retain/retry an event
		// just because annotation failed after durable raw save.
		log.Printf("[AVISO] montagem %s %d: %v", ev.Type, ev.Number, err)
	}
	log.Printf("[LOG] %s • %s #%d", session, ev.Type, ev.Number)
	ok(w)
}

func (s *server) handleInference(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "METHOD", http.StatusMethodNotAllowed)
		return
	}
	session := safeName(query(r.URL, "session", "sem_sessao"))
	body, err := readLimited(w, r, maxTextBytes)
	if err != nil {
		return
	}
	dir := s.sessionDir(session)
	p := filepath.Join(dir, "inference.csv")
	if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
		_ = atomicWrite(p, []byte("seq,raw\n"), 0o644)
	}
	_ = appendLine(p, strings.TrimSpace(string(body)))
	ok(w)
}

func (s *server) handleCrop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "METHOD", http.StatusMethodNotAllowed)
		return
	}
	session := safeName(query(r.URL, "session", "sem_sessao"))
	name := safeName(query(r.URL, "name", fmt.Sprintf("crop_%d.jpg", time.Now().UnixMilli())))
	if !strings.HasSuffix(strings.ToLower(name), ".jpg") {
		name += ".jpg"
	}
	body, err := readLimited(w, r, maxImageBytes)
	if err != nil {
		return
	}
	dir := filepath.Join(s.sessionDir(session), "crops")
	_ = os.MkdirAll(dir, 0o755)
	if err := atomicWrite(filepath.Join(dir, name), body, 0o644); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	ok(w)
}

func (s *server) handleEnd(w http.ResponseWriter, r *http.Request) {
	session := safeName(query(r.URL, "session", "sem_sessao"))
	p := filepath.Join(s.sessionDir(session), "_SESSAO_ENCERRADA.txt")
	_ = atomicWrite(p, []byte("Sessao encerrada em "+time.Now().Format("2006-01-02 15:04:05")+"\n"), 0o644)
	ok(w)
}

func (s *server) mountEvent(staging string, ev event) error {
	day := ev.Date.Format("2006-01-02")
	dayRoot := filepath.Join(s.dataRoot, day)
	if err := os.MkdirAll(dayRoot, 0o755); err != nil {
		return err
	}

	prefix := fmt.Sprintf("%s_%d_%s", normalizedType(ev.Type), ev.Number, ev.UUID)
	matches, _ := filepath.Glob(filepath.Join(staging, prefix+"*.jpg"))
	if len(matches) == 0 {
		matches, _ = filepath.Glob(filepath.Join(staging, prefix+"*.jpeg"))
	}
	if len(matches) == 0 {
		return fmt.Errorf("JPEG do evento ainda nao encontrado")
	}
	sort.Strings(matches)
	src := selectFinalImage(matches)

	var destDir, fileName string
	switch normalizedType(ev.Type) {
	case "impressao":
		g := genderFolder(ev.Gender)
		destDir = filepath.Join(dayRoot, "IMPRESSOES_MONTADAS", g)
		fileName = fmt.Sprintf("impressao_%06d_%s_%s.jpg", ev.Number, ev.Date.Format("15-04-05-000"), ev.UUID)
	case "alcance":
		destDir = filepath.Join(dayRoot, "ALCANCES_MONTADOS")
		fileName = fmt.Sprintf("alcance_%06d_%s_%s.jpg", ev.Number, ev.Date.Format("15-04-05-000"), ev.UUID)
	default:
		destDir = filepath.Join(dayRoot, "OUTROS_EVENTOS")
		fileName = fmt.Sprintf("%s_%06d_%s.jpg", safeName(ev.Type), ev.Number, ev.UUID)
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}
	dest := filepath.Join(destDir, fileName)
	if err := annotate(src, dest, ev); err != nil {
		return err
	}

	archiveDir := filepath.Join(dayRoot, "EVENTOS", fmt.Sprintf("%s_%06d_%s", strings.ToUpper(normalizedType(ev.Type)), ev.Number, ev.UUID))
	_ = os.MkdirAll(archiveDir, 0o755)
	_ = atomicWrite(filepath.Join(archiveDir, "evento.txt"), []byte(ev.RawLine+"\n"), 0o644)
	for _, p := range matches {
		_ = copyFile(p, filepath.Join(archiveDir, filepath.Base(p)))
	}

	_ = s.appendDayIndex(dayRoot, ev, dest)
	log.Printf("[MONTADO] %s", dest)
	return nil
}

func annotate(src, dst string, ev event) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	img0, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		return err
	}
	b := img0.Bounds()
	rgba := image.NewRGBA(b)
	draw.Draw(rgba, b, img0, b.Min, draw.Src)

	logicalW, logicalH := ev.FrameW, ev.FrameH
	if logicalW <= 0 {
		logicalW = b.Dx()
	}
	if logicalH <= 0 {
		logicalH = b.Dy()
	}

	bx := ev.BBox
	typ := normalizedType(ev.Type)
	rectColor := color.RGBA{0x34, 0xD3, 0x99, 0xFF}
	label := "ALCANCE | " + ev.Date.Format("2006-01-02 15:04:05.000")

	if typ == "impressao" {
		rectColor = genderColor(ev.Gender)
		label = "IMPRESSAO | " + strings.ToUpper(genderFolder(ev.Gender)) + " | " + ev.Date.Format("2006-01-02 15:04:05.000")
	} else if typ == "alcance" {
		bx = reachBodyBox(bx, logicalW, logicalH)
	}

	sx := float64(b.Dx()) / float64(maxInt(logicalW, 1))
	sy := float64(b.Dy()) / float64(maxInt(logicalH, 1))
	x0 := clampInt(int(bx.X*sx+0.5), 0, b.Dx()-2)
	y0 := clampInt(int(bx.Y*sy+0.5), 0, b.Dy()-2)
	x1 := clampInt(int((bx.X+bx.W)*sx+0.5), x0+1, b.Dx()-1)
	y1 := clampInt(int((bx.Y+bx.H)*sy+0.5), y0+1, b.Dy()-1)
	thick := clampInt((x1-x0)/45, 4, 10)
	drawRect(rgba, x0, y0, x1, y1, thick, rectColor)

	textScale := clampInt(b.Dx()/520, 2, 5)
	labelW := textWidth(label, textScale)
	labelH := 7*textScale + 8
	lx := clampInt(x0, 0, maxInt(0, b.Dx()-labelW-10))
	ly := y0 - labelH - 4
	if ly < 0 {
		ly = clampInt(y0+4, 0, maxInt(0, b.Dy()-labelH))
	}
	fillRect(rgba, lx, ly, minInt(b.Dx()-1, lx+labelW+10), minInt(b.Dy()-1, ly+labelH), color.RGBA{0, 0, 0, 220})
	drawText(rgba, lx+5, ly+4, strings.ToUpper(label), textScale, rectColor)

	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp := dst + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	encErr := jpeg.Encode(f, rgba, &jpeg.Options{Quality: 94})
	closeErr := f.Close()
	if encErr != nil {
		_ = os.Remove(tmp)
		return encErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, dst)
}

// Current Android protocol has one event bbox. For impressions that bbox is the
// current SCRFD face. For reaches it can already be a NanoDet body box, but on
// a fresh-face frame it can be the face box. If it is clearly face-shaped, use
// the exact syntheticBody geometry used by the native preview as fallback.
func reachBodyBox(b box, fw, fh int) box {
	if b.W <= 1 || b.H <= 1 {
		return b
	}
	ratio := b.H / b.W
	relativeH := b.H / float64(maxInt(fh, 1))
	if ratio >= 1.45 && relativeH >= 0.28 {
		return clipBox(b, fw, fh)
	}
	out := box{X: b.X - 1.4*b.W, Y: b.Y - 0.35*b.H, W: 3.8 * b.W, H: 6.3 * b.H}
	return clipBox(out, fw, fh)
}

func clipBox(b box, fw, fh int) box {
	maxW, maxH := float64(maxInt(fw, 1)), float64(maxInt(fh, 1))
	if b.X < 0 {
		b.W += b.X
		b.X = 0
	}
	if b.Y < 0 {
		b.H += b.Y
		b.Y = 0
	}
	if b.X+b.W > maxW {
		b.W = maxW - b.X
	}
	if b.Y+b.H > maxH {
		b.H = maxH - b.Y
	}
	if b.W < 1 {
		b.W = 1
	}
	if b.H < 1 {
		b.H = 1
	}
	return b
}

func parseEvent(line, session string) (event, error) {
	fields := parseFields(line)
	typ := fields["tipo"]
	uuid := fields["uuid"]
	if typ == "" || uuid == "" {
		return event{}, errors.New("tipo/uuid ausente")
	}
	n, _ := strconv.ParseInt(fields["numero"], 10, 64)
	d := time.Now()
	rawDate := fields["data"]
	if t, err := time.ParseInLocation("2006-01-02 15:04:05.000", rawDate, time.Local); err == nil {
		d = t
	}
	fw, fh := parseFrame(fields["frame"])
	bb := parseBox(fields["bbox"])
	return event{Type: typ, UUID: safeName(uuid), Number: n, Gender: fields["genero"], Date: d, DateRaw: rawDate, Track: fields["track"], FrameW: fw, FrameH: fh, BBox: bb, Session: session, RawLine: line}, nil
}

func parseFields(line string) map[string]string {
	m := map[string]string{}
	parts := strings.Split(line, "|")
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if i := strings.IndexByte(p, '='); i > 0 {
			m[strings.TrimSpace(p[:i])] = strings.TrimSpace(p[i+1:])
		}
	}
	return m
}

func parseFrame(v string) (int, int) {
	p := strings.Split(strings.ToLower(strings.TrimSpace(v)), "x")
	if len(p) != 2 {
		return 0, 0
	}
	w, _ := strconv.Atoi(p[0])
	h, _ := strconv.Atoi(p[1])
	return w, h
}

func parseBox(v string) box {
	p := strings.Split(v, ",")
	if len(p) != 4 {
		return box{}
	}
	vals := make([]float64, 4)
	for i := range p {
		vals[i], _ = strconv.ParseFloat(strings.TrimSpace(p[i]), 64)
	}
	return box{vals[0], vals[1], vals[2], vals[3]}
}

func (s *server) appendDayIndex(dayRoot string, ev event, mounted string) error {
	p := filepath.Join(dayRoot, "index.csv")
	newFile := false
	if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
		newFile = true
	}
	f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if newFile {
		_ = w.Write([]string{"data", "tipo", "numero", "uuid", "genero", "track", "sessao", "arquivo_montado"})
	}
	return w.Write([]string{ev.Date.Format("2006-01-02 15:04:05.000"), normalizedType(ev.Type), strconv.FormatInt(ev.Number, 10), ev.UUID, genderFolder(ev.Gender), ev.Track, ev.Session, mounted})
}

func (s *server) sessionDir(session string) string {
	p := filepath.Join(s.dataRoot, "_SESSOES", safeName(session))
	_ = os.MkdirAll(p, 0o755)
	return p
}

func selectFinalImage(paths []string) string {
	for _, p := range paths {
		if strings.Contains(strings.ToLower(filepath.Base(p)), "_final.") {
			return p
		}
	}
	// Never prefer temporal evidence as the mounted public image.
	for _, p := range paths {
		if !strings.Contains(strings.ToLower(filepath.Base(p)), "_evidencia_") {
			return p
		}
	}
	return paths[len(paths)-1]
}

func normalizedType(v string) string {
	v = strings.ToLower(strings.TrimSpace(v))
	if v == "impressão" {
		return "impressao"
	}
	return v
}

func genderFolder(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "masculino":
		return "Masculino"
	case "feminino":
		return "Feminino"
	default:
		return "Indeterminado"
	}
}

func genderColor(v string) color.RGBA {
	switch genderFolder(v) {
	case "Masculino":
		return color.RGBA{0x40, 0xC4, 0xFF, 0xFF}
	case "Feminino":
		return color.RGBA{0xFF, 0x80, 0xAB, 0xFF}
	default:
		return color.RGBA{0xFF, 0xD7, 0x40, 0xFF}
	}
}

func query(u *url.URL, key, def string) string {
	if v := u.Query().Get(key); v != "" {
		return v
	}
	return def
}

func safeName(v string) string {
	if strings.TrimSpace(v) == "" {
		return "sem_nome"
	}
	var b strings.Builder
	for _, r := range v {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '.' || r == '-' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	return b.String()
}

func readLimited(w http.ResponseWriter, r *http.Request, max int64) ([]byte, error) {
	r.Body = http.MaxBytesReader(w, r.Body, max)
	defer r.Body.Close()
	b, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "BODY_INVALIDO", http.StatusRequestEntityTooLarge)
		return nil, err
	}
	return b, nil
}

func ok(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(200)
	_, _ = io.WriteString(w, "OK")
}

func atomicWrite(p string, b []byte, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, b, mode); err != nil {
		return err
	}
	return os.Rename(tmp, p)
}

func appendLine(p, line string) error {
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(p, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	bw := bufio.NewWriter(f)
	if _, err = fmt.Fprintln(bw, line); err != nil {
		return err
	}
	return bw.Flush()
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	if err = os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	_, cp := io.Copy(out, in)
	cl := out.Close()
	if cp != nil {
		return cp
	}
	return cl
}

func localIPv4s() []string {
	var out []string
	ifs, _ := net.Interfaces()
	for _, it := range ifs {
		addrs, _ := it.Addrs()
		for _, a := range addrs {
			var ip net.IP
			switch v := a.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip != nil && ip.To4() != nil && !ip.IsLoopback() {
				out = append(out, ip.String())
			}
		}
	}
	if len(out) == 0 {
		return []string{"127.0.0.1"}
	}
	sort.Strings(out)
	return out
}

func drawRect(img *image.RGBA, x0, y0, x1, y1, t int, c color.RGBA) {
	for i := 0; i < t; i++ {
		hLine(img, x0-i, y0-i, x1+i, c)
		hLine(img, x0-i, y1+i, x1+i, c)
		vLine(img, x0-i, y0-i, y1+i, c)
		vLine(img, x1+i, y0-i, y1+i, c)
	}
}

func hLine(img *image.RGBA, x0, y, x1 int, c color.RGBA) {
	y = clampInt(y, 0, img.Bounds().Dy()-1)
	x0 = clampInt(x0, 0, img.Bounds().Dx()-1)
	x1 = clampInt(x1, 0, img.Bounds().Dx()-1)
	for x := x0; x <= x1; x++ {
		img.SetRGBA(x, y, c)
	}
}

func vLine(img *image.RGBA, x, y0, y1 int, c color.RGBA) {
	x = clampInt(x, 0, img.Bounds().Dx()-1)
	y0 = clampInt(y0, 0, img.Bounds().Dy()-1)
	y1 = clampInt(y1, 0, img.Bounds().Dy()-1)
	for y := y0; y <= y1; y++ {
		img.SetRGBA(x, y, c)
	}
}

func fillRect(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	x0 = clampInt(x0, 0, img.Bounds().Dx()-1)
	x1 = clampInt(x1, 0, img.Bounds().Dx()-1)
	y0 = clampInt(y0, 0, img.Bounds().Dy()-1)
	y1 = clampInt(y1, 0, img.Bounds().Dy()-1)
	draw.Draw(img, image.Rect(x0, y0, x1+1, y1+1), &image.Uniform{C: c}, image.Point{}, draw.Over)
}

var glyphs = map[rune][7]byte{
	' ': {0, 0, 0, 0, 0, 0, 0}, '-': {0, 0, 0, 31, 0, 0, 0}, ':': {0, 4, 4, 0, 4, 4, 0}, '.': {0, 0, 0, 0, 0, 6, 6}, '|': {4, 4, 4, 4, 4, 4, 4},
	'0': {14, 17, 19, 21, 25, 17, 14}, '1': {4, 12, 4, 4, 4, 4, 14}, '2': {14, 17, 1, 2, 4, 8, 31}, '3': {30, 1, 1, 14, 1, 1, 30}, '4': {2, 6, 10, 18, 31, 2, 2}, '5': {31, 16, 16, 30, 1, 1, 30}, '6': {14, 16, 16, 30, 17, 17, 14}, '7': {31, 1, 2, 4, 8, 8, 8}, '8': {14, 17, 17, 14, 17, 17, 14}, '9': {14, 17, 17, 15, 1, 1, 14},
	'A': {14, 17, 17, 31, 17, 17, 17}, 'B': {30, 17, 17, 30, 17, 17, 30}, 'C': {14, 17, 16, 16, 16, 17, 14}, 'D': {30, 17, 17, 17, 17, 17, 30}, 'E': {31, 16, 16, 30, 16, 16, 31}, 'F': {31, 16, 16, 30, 16, 16, 16}, 'G': {14, 17, 16, 23, 17, 17, 15}, 'H': {17, 17, 17, 31, 17, 17, 17}, 'I': {14, 4, 4, 4, 4, 4, 14}, 'J': {7, 2, 2, 2, 18, 18, 12}, 'K': {17, 18, 20, 24, 20, 18, 17}, 'L': {16, 16, 16, 16, 16, 16, 31}, 'M': {17, 27, 21, 21, 17, 17, 17}, 'N': {17, 25, 21, 19, 17, 17, 17}, 'O': {14, 17, 17, 17, 17, 17, 14}, 'P': {30, 17, 17, 30, 16, 16, 16}, 'Q': {14, 17, 17, 17, 21, 18, 13}, 'R': {30, 17, 17, 30, 20, 18, 17}, 'S': {15, 16, 16, 14, 1, 1, 30}, 'T': {31, 4, 4, 4, 4, 4, 4}, 'U': {17, 17, 17, 17, 17, 17, 14}, 'V': {17, 17, 17, 17, 17, 10, 4}, 'W': {17, 17, 17, 21, 21, 21, 10}, 'X': {17, 17, 10, 4, 10, 17, 17}, 'Y': {17, 17, 10, 4, 4, 4, 4}, 'Z': {31, 1, 2, 4, 8, 16, 31},
}

func drawText(img *image.RGBA, x, y int, s string, scale int, c color.RGBA) {
	cx := x
	for _, r := range s {
		g, ok := glyphs[r]
		if !ok {
			g = glyphs[' ']
		}
		for row := 0; row < 7; row++ {
			for col := 0; col < 5; col++ {
				if (g[row] & (1 << (4 - col))) != 0 {
					fillRect(img, cx+col*scale, y+row*scale, cx+(col+1)*scale-1, y+(row+1)*scale-1, c)
				}
			}
		}
		cx += 6 * scale
	}
}

func textWidth(s string, scale int) int { return len([]rune(s)) * 6 * scale }

func clampInt(v, lo, hi int) int {
	if hi < lo {
		return lo
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
