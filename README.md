# Spurerkennung

## Konzept
### Anforderungen
- Zusatzaufgaben:
	- Challenge Video 
	- KI - Ansatz 
- Wenn nur KI Ansatz und Android = 1.0 dann die beiden machen 

- **Segmentierung**: schränken Sie das Bild auf den Bereich ein, in dem sich die Spurmarkierungen befinden

- **Vorverarbeitung**: führen Sie eine Kamerakalibrierung (für Udacity-Bildquellen) und die Perspektivtransformation durch

- **Farbräume, Histogramme**: erkennen Sie die Spurmarkierungen in den Farben der angegebenen Quellen. Sofern weitere Spurmarkierungen auf dem Bild gefunden werden, müssen diejenigen Spurmarkierungen priorisiert werden, die die eigene Fahrspur begrenzen

- **Allgemeines**: Die Verarbeitung von Bildern muss in Echtzeit stattfinden --> Ziel: > 20 FPS

- **Allgemeines**: Beschleunigen Sie die Verarbeitung durch weitere Maßnahmen weitere Maßnahmen überlegen (bspw. Erkennung der Spurmarkierung in den ersten Frames, Tracking der Spurmarkierung in weiteren Frames solange, bis sich Spurmarkierungspositionen zu stark 

- **Minimal**: relevante Spurmarkierungen werden im Video "project_video" durchgehend erkannt

- **Zusatz**: relevante Spurmarkierungen werden im Video "challenge_video" oder "harder_challenge_video" durchgehend erkannt

### Dokumenation
- Hauptsächlich über PowerPoint
- Sonstiges in Readme oder andere [[Markdown]] files

### Sonstiges 
[Vortrainierte KI](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2 "https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2")
