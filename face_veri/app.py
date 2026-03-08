from face_system import NAIFFaceSystem

face_ai = NAIFFaceSystem()

choice = input("Register new person? (y/n): ")

if choice.lower() == "y":
    name = input("Enter name: ")
    face_ai.register_person(name)

print("Starting live recognition...")
names = face_ai.get_current_names()

print("People detected:", names)

if "Nitish" in names:
    print("Nitish is sitting next to NAIF 🤖")