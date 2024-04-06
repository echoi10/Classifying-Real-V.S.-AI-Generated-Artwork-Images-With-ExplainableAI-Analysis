import json
import requests
import base64
import csv
import time

# run = 80
prompts = []

with open('all_prompts.csv', newline='') as csvfile:
    linereader = csv.reader(csvfile, delimiter=' ')
    for row in linereader:
        prompts.append(' '.join(row))

print("prompts read")

def submit_batch():
    run = 21 # change

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": ""
    }
    
    payload = { "requests": [] }

    for i in range(10000, len(prompts)): #change

        if ((i % 500 == 0 and i != 10000) or i == len(prompts) - 1):
            url = "https://api.edenai.run/v2/image/generation/batch/final" + str(run) + "/"
            print("test", i, url)
            response = requests.post(url, json=payload, headers=headers)

            print("job " + str(run) + "submitted")
            print(response.text)
            
            run += 1
            payload = { "requests": [] }

            if (i % 2500 == 0):
                time.sleep(90 * 60) #1.5 hour
                print(i, "sleeping for 1.5 hour")

        payload["requests"].append({
            "response_as_dict": True,
            "attributes_as_list": False,
            "show_original_response": False,
            "resolution": "1024x1024",
            "num_images": 1,
            "text": prompts[i],
            "providers": "openai"
        })  

run = 55 # replace w last run number

def get_batch_results():
    j = 7500
    k = "16_3"
   
    url = "https://api.edenai.run/v2/image/generation/batch/final" + str(k) + "/"

    headers = {
        "accept": "application/json",
        "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMzhlZjBkZDgtMDUwYS00OWM4LTg1ZDgtYmVlOTY0NzViNzFhIiwidHlwZSI6ImFwaV90b2tlbiJ9.78WZbGBe455paSwHbuUOu4AbWlc6d895_tlJk1J1WIw"
    }

    try:
        response = requests.get(url, headers=headers)
        res = response.json()
    except:
        print("final" + str(k) + "failed")
        # j += 500

    image_path = "./outputs"

    # print(res)
    print("final" + str(k))
    if (res and "requests" in res):
        for request in res["requests"]:
            if (request["response"] != None and request["status"] == "succeeded"):
                if "items" in request["response"]["openai"]:
                    image = request["response"]["openai"]["items"][0]["image"]
                    decoded_image = base64.b64decode(image)
                    filename = image_path + "/" + prompts[j].replace(' ', '_') + ".png"
                    with open(filename, 'wb') as f:
                        f.write(decoded_image)
                    j += 1
                    print("first", j)
                    print("last page",res["last_page"])
                    print("per page",res["per_page"])
                else:
                    j += 1
                    continue
            else:
                j += 1
                continue

    if (res and "last_page" in res):
        for i in range(2, res["last_page"]+1):
            url = "https://api.edenai.run/v2/image/generation/batch/final" + str(k) + "/?page=" + str(i)
            try:
                response = requests.get(url, headers=headers)
                res = response.json()
            except:
                print("final" + str(k) + "failed")
                j += 50
                continue

            if (res and "requests" in res):
                for request in res["requests"]:
                    if (request["response"] != None and request["status"] == "succeeded"):
                        if "items" in request["response"]["openai"]:
                            image = request["response"]["openai"]["items"][0]["image"]
                            decoded_image = base64.b64decode(image)
                            filename = image_path + "/" + prompts[j].replace(' ', '_') + ".png"
                            with open(filename, 'wb') as f:
                                f.write(decoded_image)
                            j += 1
                            print("next",j)
                        else:
                            j += 1
                            continue
                    else:
                            j += 1
                            continue

# submit_batch()
get_batch_results()

