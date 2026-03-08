from face_system import NAIFFaceSystem

face_ai = NAIFFaceSystem()

names = face_ai.get_live_names()

print("Detected:", names)