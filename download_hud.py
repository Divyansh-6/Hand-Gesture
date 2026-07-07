import urllib.request
url = 'https://www.pngmart.com/files/10/Iron-Man-HUD-Transparent-PNG.png'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open('assets/iron_man_hud.png', 'wb') as out_file:
    data = response.read()
    out_file.write(data)
print("Downloaded successfully!")
