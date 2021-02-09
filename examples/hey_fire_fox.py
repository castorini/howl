from howl.client import HowlClient

"""
An example script for running the pre-trained howl model for Firefox Voice. 

Usage:
    python -m examples.hey_fire_fox
"""

def hello_callback(detected_words):
    print("Detected: {}".format(detected_words))

client = HowlClient()
client.from_pretrained("hey_fire_fox", force_reload=True)
client.add_listener(hello_callback)
client.start().join()
