# NAIF – Neural Autonomous Intelligence Framework

NAIF (Neural Autonomous Intelligence Framework) is an offline multimodal AI assistant that integrates conversational AI, computer vision, healthcare analysis, coding assistance, and smart security into a single intelligent system.

The system combines Large Language Models, speech technologies, computer vision, and retrieval-based knowledge systems to create an AI assistant capable of interacting with users through voice, text, and camera input.

---

## Project Overview

NAIF is designed to function primarily offline, reducing dependency on cloud-based AI services while maintaining privacy and fast response times.

The framework integrates multiple AI modules that allow it to perform tasks such as:

- Conversational AI interaction
- Programming assistance
- Medical prescription analysis
- Face recognition based security

The goal of NAIF is to build a versatile AI system capable of assisting users in real-world environments.

---

## Key Features

### 1. AI Chatbot Assistant

The core component of NAIF is an intelligent chatbot capable of natural language conversation and contextual responses.

Features include:

- Natural conversation using Large Language Models
- Voice interaction using Speech-to-Text and Text-to-Speech
- Memory system to store important user information
- Camera vision integration for contextual interaction
- Internet search capability when network is available

---

### 2. Coding Assistant

NAIF includes a coding assistant that helps users with programming tasks.

Capabilities include:

- Code generation
- Code explanation
- Debugging assistance
- Support for multiple programming languages

The coding assistant runs through a Flask-based interface powered by a code-focused language model.

---

### 3. Medical Assistant

NAIF includes a healthcare assistance module capable of analyzing medical prescriptions.

Capabilities include:

- Prescription image reading
- Optical Character Recognition (OCR) for handwritten text
- Medicine name detection
- Fuzzy matching with a large medicine dataset
- Healthcare information retrieval

The system uses OCR and similarity matching techniques to identify medicine names even when handwriting is unclear.

---

### 4. Smart Door Lock System

NAIF also includes a computer-vision-based security system.

Main features include:

- Face recognition for authorized access
- Detection of unknown individuals
- Alarm trigger for unauthorized entry
- Camera-based image capture
- Gesture-based interaction

This module provides AI-based security using real-time vision processing.

---

## Technologies Used

### Programming Language
- Python

### AI Models
- Gemma 3 (4B) – Conversational AI
- Qwen2.5 Coder (7B) – Coding Assistant

### AI Frameworks
- Ollama
- LangChain
- LangGraph

### Speech Technologies
- Vosk (Speech-to-Text)
- Text-to-Speech (TTS)

### Computer Vision
- OpenCV

### Web Technologies
- Flask
- HTML
- CSS
- JavaScript

### AI Techniques
- Retrieval Augmented Generation (RAG)
- Machine Learning
- Fuzzy Matching

### Data Processing
- Pandas
- NumPy
- RapidFuzz

### OCR
- TrOCR Transformer Model

---

## System Architecture

User interaction occurs through:

- Voice input
- Text interface
- Camera vision

The input is processed by the **NAIF AI Core**, which routes the request to the appropriate module:

- Chatbot Assistant
- Coding Assistant
- Medical Assistant
- Smart Security System

The response is delivered through:

- Text output
- Voice output
- Visual interface

---

## Developer

Nitish Kumar Sharma

---

## Project Type

Artificial Intelligence System  
Computer Vision + Voice AI + LLM Integration

---

## License

This project is intended for educational and research purposes.
