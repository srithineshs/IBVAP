import socket
import hashlib
import re

def send_rtsp(ip, port, req):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((ip, port))
    s.sendall(req.encode('utf-8'))
    resp = s.recv(4096).decode('utf-8', errors='ignore')
    s.close()
    return resp

ip = "10.245.106.240"
port = 554

# 1. Send OPTIONS
options_req = f"OPTIONS rtsp://{ip}:{port}/Streaming/Channels/101 RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: Antigravity\r\n\r\n"
print("--- 1. OPTIONS Request ---")
resp1 = send_rtsp(ip, port, options_req)
print(resp1)

# 2. Send DESCRIBE (unauthenticated to get 401 & Nonce)
describe_req = f"DESCRIBE rtsp://{ip}:{port}/Streaming/Channels/101 RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\nUser-Agent: Antigravity\r\n\r\n"
print("--- 2. DESCRIBE Request ---")
resp2 = send_rtsp(ip, port, describe_req)
print(resp2)

# Extract Digest auth parameters
realm_match = re.search(r'realm="([^"]+)"', resp2)
nonce_match = re.search(r'nonce="([^"]+)"', resp2)

if realm_match and nonce_match:
    realm = realm_match.group(1)
    nonce = nonce_match.group(1)
    print(f"\n[+] Realm: {realm}, Nonce: {nonce}")
    
    # Calculate Digest Response for admin:admin@123
    username = "admin"
    password = "admin@123"
    uri = f"rtsp://{ip}:{port}/Streaming/Channels/101"
    
    # HA1 = MD5(username:realm:password)
    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode('utf-8')).hexdigest()
    # HA2 = MD5(method:uri)
    ha2 = hashlib.md5(f"DESCRIBE:{uri}".encode('utf-8')).hexdigest()
    # Response = MD5(HA1:nonce:HA2)
    auth_resp = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode('utf-8')).hexdigest()
    
    auth_header = f'Digest username="{username}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{auth_resp}"'
    
    describe_auth_req = f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: 3\r\nAuthorization: {auth_header}\r\nAccept: application/sdp\r\nUser-Agent: Antigravity\r\n\r\n"
    print("\n--- 3. Authenticated DESCRIBE Request ---")
    resp3 = send_rtsp(ip, port, describe_auth_req)
    print(resp3)
