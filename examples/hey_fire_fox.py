import os
from howl.client import HowlClient

def hello_callback(detected_words):
    print("Detected: {}".format(detected_words))

client = HowlClient()
client.from_pretrained("hey_fire_fox", force_reload=True)
client.add_listener(hello_callback)
client.start().join()
