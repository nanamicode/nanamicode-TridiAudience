# Tridi Collector Server — Linux

Servidor compatível com o protocolo existente do TridiAudience:

- discovery UDP `8790`: `TRIDI_COLLECTOR_DISCOVER_V1` → `TRIDI_COLLECTOR_HERE|8791`
- uploads HTTP `8791`: `/image`, `/log`, `/inference`, `/crop`, `/end`, `/ping`
- nenhum IP precisa ser configurado no totem

## Organização

- `DADOS_TVBOX/AAAA-MM-DD/ALCANCES_MONTADOS/`
  - um JPEG montado por alcance
  - somente o retângulo corporal
  - data/hora desenhada na imagem
- `DADOS_TVBOX/AAAA-MM-DD/IMPRESSOES_MONTADAS/Masculino|Feminino|Indeterminado/`
  - um JPEG montado por impressão
  - somente o retângulo facial
  - azul = masculino, rosa = feminino, amarelo = indeterminado
  - data/hora desenhada na imagem
- `DADOS_TVBOX/AAAA-MM-DD/EVENTOS/`
  - JPEGs originais e `evento.txt`
- `DADOS_TVBOX/AAAA-MM-DD/index.csv`
  - índice diário
- `DADOS_TVBOX/_SESSOES/`
  - staging bruto compatível com o coletor antigo

O alcance usa a caixa corporal recebida. Quando o protocolo legado entrega uma
caixa claramente facial, o servidor reconstrói o mesmo `syntheticBody` usado
pelo core nativo para não desenhar retângulo de rosto em ALCANCES_MONTADOS.

## Executar

```bash
chmod +x TridiCollectorServer INICIAR_SERVIDOR.sh
./INICIAR_SERVIDOR.sh
```

Ou marque `TridiCollectorServer.desktop` como executável e abra-o pelo ambiente gráfico.

Se houver firewall, libere UDP 8790 e TCP 8791.

O binário Linux é estático e não depende de Python, PowerShell, ImageMagick ou bibliotecas externas.
