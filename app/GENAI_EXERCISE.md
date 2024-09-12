# Aufgabenbeschreibung

## Allgemeine Infos

- **Entwicklungsumgebung**: Implementiere deine Lösung im `app` Verzeichnis des Repositories. Dies gewährleistet eine saubere Struktur und einfache Deployment-Optionen.
- **Prototyping**: Beginne mit einem Jupyter Notebook, um schnell zu prototypen. Dies ermöglicht es dir, Ideen schneller zu testen und zu iterieren.
- **Modularisierung**: Achte darauf, die Funktionalität sinnvoll in Methoden und Files zu trennen. Dies erleichtert die Wartung und das Testing der Anwendung.

## Aufgabe 1: ChatUI

- **Ziel**: Erstelle eine interaktive ChatUI mittels Streamlit. Diese UI wird die Hauptoberfläche für die Benutzerinteraktion sein. Ectl helfen dir diese Vorlagen: <https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps>
- **Technologieauswahl**: Nutze ein Package deiner Wahl (Haystack, LangChain oder LlamaIndex) um das Ollama Modell an die ChatUI anzubinden. Überprüfe die Dokumentation der Packages für eine optimale Integration.
- **Performance**: Streaming der Daten ist optional. Überlege, ob Streaming notwendig ist für deine Anwendung oder ob eine einfachere Implementierung für den Anfang besser ist.

## Aufgabe 2: Indexing

- **Datenquelle**: Verwende das Tesla Manual im `data` Pfad. Dieses PDF dient als Grundlage für die Erstellung der Datenbasis.
- **PDF-Verarbeitung**: Transformiere das PDF in ein geeignetes Dokumentenformat, das für Textanalysen optimiert ist. Tools wie `haystack` oder `llamaindex` oder `langchain` bieten hier nützliche Werkzeuge.
- **Chunking**: Entwickle eine Chunking-Strategie, die es ermöglicht, die Bedienungsanleitung in sinnvolle Abschnitte zu unterteilen. Dies erleichtert die Indexierung und Suche von Informationen.
- **Speicherung**: Nutze einen InMemory Store, um die erstellten Dokumente temporär zu speichern. Dies beschleunigt den Zugriff während der Entwicklungsphase.

Du kannst die Indexing Pipeline auch in einem Notebook konzipieren. Sie könnte aus folgenden Teilen bestehen:

- **Einlesen des Dokuments**
- **Cleaning**
- **Chunking**
- **Embedding Generation**
- **Speichern im Vector Storage**

## Aufgabe 3: Embeddings

- **Modellauswahl**: Finde ein Modell von Sentence Transformers, das effektiv Embeddings für die Sprache deines Manuals generieren kann. Beachte dabei die sprachlichen Besonderheiten des Manuals.
- **Integration**: Integriere das gewählte Modell in deine Indexingpipeline, um neben dem reinen Text auch Embeddings zu speichern. Dies verbessert die Qualität der Suchergebnisse.
- **Ressourcennutzung**: Achte auf die Ressourcenanforderungen des Embedding-Modells, um eine effiziente Verarbeitung sicherzustellen.

## Aufgabe 4: Retrieval Pipeline

- **Frage-Antwort-System**: Entwickle eine Pipeline, die Benutzerfragen entgegennimmt und relevante Chunks aus dem Index zurückgibt. Dies bildet die Grundlage für ein effektives Information Retrieval System.
- **Sprachverarbeitung**: Optional für Tag 1: Nutze ein Large Language Model (LLM), um aus den gefundenen Chunks eine kohärente Antwort in natürlicher Sprache zu generieren. Dies erhöht die Benutzerfreundlichkeit des Systems.

Eine Retrieving Pipeline könnte bestehen aus:

- **Query**
- **Embedder**
- **Retriever**
- **Prompt**
- **Generator**
