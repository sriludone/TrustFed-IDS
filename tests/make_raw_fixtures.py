import os, sys, numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "fixtures_raw"
rng = np.random.default_rng(0)

def blobs(labels, sizes, d, heavy=True):
    X, y = [], []
    for k, (lab, n) in enumerate(zip(labels, sizes)):
        mu = rng.normal(0, 1.2, d)
        Z = rng.normal(mu, 1.0, (n, d))
        X.append(np.exp(Z) if heavy else Z); y += [lab] * n
    return np.vstack(X), np.array(y)


labs = ["BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP-Patator", "SSH-Patator",
        "DoS slowloris", "DoS Slowhttptest", "Bot", "Web Attack \u2013 Brute Force",
        "Web Attack \u2013 XSS", "Infiltration", "Web Attack \u2013 Sql Injection", "Heartbleed"]
sizes = [6000, 1500, 1200, 1000, 400, 350, 300, 300, 300, 250, 200, 150, 30, 20, 10]
X, y = blobs(labs, sizes, 20)
cols = [" Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Total Length of Fwd Packets",
        " Fwd Packet Length Max", " Fwd Packet Length Min", "Flow Bytes/s", " Flow Packets/s",
        " Flow IAT Mean", " Fwd IAT Std", " Bwd Packets/s", " Min Packet Length", " Max Packet Length",
        " Packet Length Mean", "FIN Flag Count", " SYN Flag Count", " ACK Flag Count", "Init_Win_bytes_forward",
        " Init_Win_bytes_backward", " Fwd Header Length"]
df = pd.DataFrame(X, columns=cols)
df.insert(0, "Flow ID", [f"10.0.0.{i%250}-192.168.1.{i%200}-{i}" for i in range(len(df))])
df.insert(1, " Source IP", "192.168.10.5"); df.insert(2, " Source Port", rng.integers(1024, 65535, len(df)))
df.insert(3, " Destination IP", "192.168.10.50"); df.insert(4, " Destination Port", rng.integers(1, 1024, len(df)))
df.insert(5, " Timestamp", "3/7/2017 8:55"); df[" Fwd Header Length.1"] = df[" Fwd Header Length"]
df.loc[::97, "Flow Bytes/s"] = np.inf
df[" Label"] = y
df = pd.concat([df, df.sample(800, random_state=1)])
os.makedirs(f"{R}/CICIDS2017", exist_ok=True)
sh = df.sample(frac=1, random_state=2); n3 = len(sh)//3
for i, part in enumerate([sh.iloc[:n3], sh.iloc[n3:2*n3], sh.iloc[2*n3:]]):
    part.to_csv(f"{R}/CICIDS2017/day{i}.pcap_ISCX.csv", index=False, encoding="cp1252")


from trustfed_fam import FAM
labs = list(FAM); sizes = [3000 if l == "BenignTraffic" else (1500 if l.startswith("DDoS") else 120) for l in labs]
sizes[labs.index("Uploading_Attack")] = 40; sizes[labs.index("Recon-PingSweep")] = 60
cc = ["flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
      "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
      "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
      "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
      "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
      "Variance", "Weight"]
X, y = blobs(labs, sizes, len(cc))
df = pd.DataFrame(X, columns=cc); df["label"] = y
os.makedirs(f"{R}/CICIoT2023", exist_ok=True)
sh = df.sample(frac=1, random_state=3); n4 = len(sh)//4
for i, part in enumerate([sh.iloc[k*n4:(k+1)*n4 if k<3 else None] for k in range(4)]):
    part.to_csv(f"{R}/CICIoT2023/part-{i:05d}.csv", index=False)


labs = ["Normal", "DDoS_UDP", "DDoS_ICMP", "SQL_injection", "Password", "Vulnerability_scanner", "DDoS_TCP",
        "DDoS_HTTP", "Uploading", "Backdoor", "Port_Scanning", "XSS", "Ransomware", "MITM", "Fingerprinting"]
sizes = [6000, 1200, 1100, 500, 500, 500, 500, 500, 400, 300, 300, 300, 200, 60, 50]
num = ["arp.opcode", "icmp.seq_le", "http.content_length", "tcp.ack", "tcp.connection.fin", "tcp.connection.syn",
       "tcp.flags", "tcp.len", "tcp.seq", "udp.time_delta", "dns.qry.qu", "dns.qry.type", "mqtt.len",
       "mqtt.msgtype", "mqtt.topic_len", "mbtcp.len", "mbtcp.unit_id"]
X, y = blobs(labs, sizes, len(num))
df = pd.DataFrame(X, columns=num)
idx = {l: i for i, l in enumerate(labs)}
df["http.request.method"] = [["0", "GET", "POST"][idx[v] % 3] for v in y]
df["mqtt.topic"] = [["0", "Temperature_and_Humidity"][idx[v] % 2] for v in y]
df["mqtt.protoname"] = [["0", "MQTT"][(idx[v] // 2) % 2] for v in y]
for c in ["http.referer", "http.request.version", "dns.qry.name.len", "mqtt.conack.flags"]:
    df[c] = "0"
df.insert(0, "frame.time", "2021 11:44:10.08"); df.insert(1, "ip.src_host", "192.168.0.128")
df.insert(2, "ip.dst_host", "192.168.0.101"); df["tcp.srcport"] = rng.integers(1, 65535, len(df))
df["tcp.dstport"] = 1883; df["udp.port"] = 0; df["udp.stream"] = np.arange(len(df))
df["tcp.payload"] = "0"; df["tcp.options"] = "0"; df["mqtt.msg"] = "0"; df["dns.qry.name"] = "0"
df["Attack_label"] = (y != "Normal").astype(int); df["Attack_type"] = y
df = pd.concat([df, df.sample(500, random_state=4)])
os.makedirs(f"{R}/Edge-IIoTset/Selected dataset for ML and DL", exist_ok=True)
df.to_csv(f"{R}/Edge-IIoTset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv", index=False)
print("fixtures written to", R)
