# Spurerkennung

Dieses Projekt beinhaltet die Prüfungsleistung für den Kurs Digitale Bildverarbeitung an der Dualen Hochschule Bade-Württemberg. Aufgabe ist es eine Spurerkennung für ein Video zu erstellen. Bonusaufgaben können zudem ergänzt werden. Für genauere Anforderung siehe [Task.pdf](Task.pdf).

## Aufbau des Projekts

```Text
|- config: Konfiguration für jedes Video
|- Documentation: Dokumentation des Projekts
    |- Images: Bilder für die Dokumentation
    |- Videos: Videos der endgültigen Version
|- img: Bilder und Videos für das Projekt
|- src: Performante Python Implementierung
    |- model: KI-Modell
calibration.calib: Kamera Kalibrierung
Task.pdf: Aufgabenstellung
```

## Gewählte Vorgehensweise

Das Ziel ist, verschiedene Ansätze für die Spurerkennung zu entwickeln und sie zu vergleichen. Dazu gehört, dass das Bild zuerst entzerrt und gegebenenfalls in Vogelperspektive transformiert wird. Danach wurde sich für folgenden weiteren Ansätze entschieden:

1. Hough Transformation
2. Sliding Windows
3. KI-Modell

Der Gedanke hinter dieser gewählten Reihenfolge war, zuerst mit etwas simplerem anzufangen. Zudem wurde die Hough Transformation in der Vorlesung schon behandelt und bot somit einen guten Einstieg. Durch eine Internet-Recherche wurde das Modell der Sliding Windows entdeckt. Obwohl es zunächst prototypisch implementiert war, funktionierte es sehr gut und hat den Hough Ansatz in seiner Genauigkeit bei Weitem übertroffen (siehe Bilder: Hough oben, Sliding Windows unten).
![Hough-Transformation](Documentation/Images/hough_draw_polyline.png)
![Sliding-Window](Documentation/Images/sliding_windows_road.png)

Wie in den beiden Bildern zu sehen, findet der Sliding Window Ansatz die Spur deutlich korrekter und kann weiter in die Ferne schauen. Deswegen wurde sich dafür entschlossen, den Ansatz vollständig zu implementieren und als Erweiterung zu verwenden.

Im Allgemeinen wurde das Entzerren, die Vogelperspektive, die Hough Transformation und der Sliding Window Ansatz zuerst prototypisch in [Prototyping.ipynb](Documentation/Prototyping.ipynb) implementiert. Aufgrund von Unübersichtlichkeit und performance einbusen im Jupyter Notebook, wurde sich danach entschieden die Prototypen in Python nativ zu implementieren. Ab diesem Punkt wurden die Prototypen nicht weiter im Notebook entwickelt. Um die Spurerkennung einfacher für die Videos konfigurieren zu können, wurde danach eine Architektur für die Konfiguration und den Ablauf des Programms entwickelt:

![Klassen-Diagramm](Documentation/Images/class_diagram.drawio.png) <!--TODO: KI-Ansatz mit rein nehmen (V: Benni)-->

Es wurde sich explizit dafür entschieden, dass es eine Main-Klasse und für jeden Ansatz eine eigene Klasse gibt. Die Main-Klasse ist dabei für das Starten des Videos und Aufrufen der Unterklassen zuständig. Die einzelnen Ansätze funktionieren unabhängig voneinander und benutzten Hilfsklassen, um gemeinsame Funktionen auszuführen. Außerdem gibt es für die Transformation und die Kamerakalibrierung jeweils eine eigene Klasse.

Als vorletztes wurde die Spurerkennung auf das Challenge Video ausgeweitet und konfiguriert, um eine Generalisierbarkeit der Spurerkennung zu test. Das Ergebnis ist, dass beide Ansätze (Hough und Sliding Window) auf dem Projekt und Challenge Video mit über 20 FPS funktionieren.

Um eine noch bessere Genauigkeit zu erzielen, wurde sich mit einem KI-Ansatz auseinander gesetzt. Zuerst sollte dieser komplett von Scratch entwickelt werden. Jedoch wurde sich, aufgrund von schwacher Performance des selbst gebautenModells und fehlender Daten, für ein vor trainiertes Modell entschieden. Dieses erzielte bereits nach erfolgreicher Einbindung in den Prototyp akzeptable Ergebnisse. Nach weiteren Modifizierungen der Input und Output Verarbeitung wurden gute Ergebnisse erzielt.

## Zusatzfunktionen für Bonusaufgaben


### Quellen und Code-Snippets

Diese Referenzen wurden als Inspiration für die dargestellten Lösungen verwendet. Teilweise wurden einzelne Code snippets verwendet.

- <https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0> (Segmentation and Hugh Transformation)
- <https://towardsdatascience.com/a-deep-dive-into-lane-detection-with-hough-transform-8f90fdd1322f> (Segmentation and Hugh Transformation)
- <https://github.com/AndrikSeeger/Lane_Detection> (Alleskönner)
- <https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html> (Image perspective transformation)
- <https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143> (Image perspective transformation)
- <https://github.com/tatsuyah/Lane-Lines-Detection-Python-OpenCV> (Sliding Windows generation and calculation)
- <https://kushalbkusram.medium.com/advanced-lane-detection-fd39572cfe91> (Sliding Window understand how they work)
- [https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2]("https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2") (Vortrainierte Modell)
