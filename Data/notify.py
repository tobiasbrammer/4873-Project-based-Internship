import http.client, urllib

def notify(message):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
                 urllib.parse.urlencode({
                     "token": "aqyvqcjojwp9zbw8wy7ruhe1se3cem",
                     "user": "uyz9r6uc12vrzoa67ewry4i77k73n4",
                     "message": message,
                 }), {"Content-type": "application/x-www-form-urlencoded"})
    conn.getresponse()