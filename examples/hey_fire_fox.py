from howl.client import HowlClient

def hello_callback(detected_words):
    print("Detected: {}".format(detected_words))

client = HowlClient()
client.from_pretrained("hey_fire_fox", force_reload=False)
client.add_listener(hello_callback)
client.start().join()
