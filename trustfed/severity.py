import re

BENIGN = re.compile(r"^(benign|normal|benigntraffic|benign_traffic)$", re.I)
LOW = re.compile(r"(portscan|port_scan|scan|recon|fingerprint|probe|sweep|vulnerability|"
                 r"pingsweep|hostdiscovery)", re.I)


def severity_of(name: str) -> int:
    n = str(name).strip()
    if BENIGN.match(n.replace(" ", "")):
        return 0
    if LOW.search(n):
        return 1
    return 2


def severity_map(class_names):
    return [severity_of(c) for c in class_names]


def benign_index(class_names):
    for i, c in enumerate(class_names):
        if severity_of(c) == 0:
            return i
    return None
