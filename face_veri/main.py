import sys
from face_system import NAIFFaceSystem

face_ai = NAIFFaceSystem()

if len(sys.argv) < 2:
    print("Usage:")
    print("  python main.py register <Name>")
    print("  python main.py verify")
    sys.exit()

mode = sys.argv[1]

if mode == "register":
    if len(sys.argv) < 3:
        print("Please provide name.")
    else:
        name = sys.argv[2]
        face_ai.register_person(name)

elif mode == "verify":
    face_ai.live_camera_verification()

else:
    print("Invalid command")