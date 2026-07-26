"""ONVIF WS-Discovery over UDP multicast.

ONVIF devices announce themselves via WS-Discovery: we send a SOAP ``Probe``
to the standard multicast group (239.255.255.250:3702) and parse the
``XAddrs`` field out of every reply to learn each camera's real IP and ONVIF
service port -- no need to know whether it listens on 80, 2020, etc.

macOS gotcha: a bare ``sendto`` to the multicast group lets the kernel choose
the egress interface. With VPN tunnels (utun*) active it intermittently routes
the probe into a tunnel and fails with ``[Errno 65] No route to host``. Setting
``IP_MULTICAST_IF`` is only a *preference* -- macOS still consults the routing
table for 239.255.255.250, so a VPN that owns that route can still break it. The
durable fix is macOS ``IP_BOUND_IF``, which pins the socket to an interface
*below* the routing table and cannot be hijacked. We resolve each real LAN
interface and probe from every candidate, so discovery works regardless of VPNs
or which Wi-Fi/Ethernet interface the camera is on.

Only Python's standard library is used, so this stays dependency-free.
"""

from __future__ import annotations

import errno
import re
import socket
import subprocess
import uuid
from typing import NamedTuple, TypedDict

MULTICAST_GROUP = "239.255.255.250"
MULTICAST_PORT = 3702

_LOCAL_NETWORK_HINT = (
    "macOS Local Network Privacy is blocking this app. Grant your terminal "
    "(Terminal / iTerm / Warp / VS Code) access under System Settings -> "
    "Privacy & Security -> Local Network, then fully quit and reopen it. "
    "(Running from inside an already-approved app such as Claude works; a "
    "terminal that lacks the permission returns 'No route to host'.)"
)

# macOS-only socket option: bind a socket's traffic to a specific interface
# index, overriding the routing table. Not exposed as a constant by Python.
_IP_BOUND_IF = 25

# Interface name prefixes that are never the camera's LAN (loopback, VPN
# tunnels, 6-to-4, etc.).
_SKIP_IFACE_PREFIXES = ("lo", "utun", "gif", "stf", "ipsec", "ppp", "awdl", "llw")


class Interface(NamedTuple):
    name: str | None   # e.g. "en1"; None when only an IP override is known
    ip: str            # IPv4 address to bind/send from
    index: int | None  # if_nametoindex value for IP_BOUND_IF, when known


class DiscoveredCamera(TypedDict):
    ip: str
    port: int
    xaddr: str


def _primary_lan_ip() -> str | None:
    """Best-effort source IP of the interface that reaches the LAN gateway.

    Uses a connected UDP socket, which makes the OS resolve the outbound route
    without sending any packet. Returns ``None`` if it can't be determined.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Any routable address works; nothing is actually sent.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _iface_ipv4(name: str) -> str | None:
    """IPv4 of a named interface via macOS ``ipconfig getifaddr`` (or None)."""
    try:
        out = subprocess.run(
            ["ipconfig", "getifaddr", name],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _candidate_interfaces(bind_ip: str | None = None) -> list[Interface]:
    """Real LAN interfaces to probe from, primary (default-route) first.

    Enumerates interfaces via ``if_nameindex`` and keeps those with a usable
    IPv4, skipping loopback/VPN/link-local. Falls back to the connect-trick IP
    if enumeration yields nothing (e.g. non-macOS hosts).
    """
    primary_ip = _primary_lan_ip()

    if bind_ip:
        # Honor the override; attach an interface index if we can find it.
        for index, name in socket.if_nameindex():
            if _iface_ipv4(name) == bind_ip:
                return [Interface(name, bind_ip, index)]
        return [Interface(None, bind_ip, None)]

    interfaces: list[Interface] = []
    seen_ips: set[str] = set()
    try:
        name_index = socket.if_nameindex()
    except OSError:
        name_index = []

    for index, name in name_index:
        if name.startswith(_SKIP_IFACE_PREFIXES):
            continue
        ip = _iface_ipv4(name)
        if not ip or ip.startswith(("127.", "169.254.")) or ip in seen_ips:
            continue
        seen_ips.add(ip)
        interfaces.append(Interface(name, ip, index))

    # Fallback (non-macOS or empty): route-based primary IP, no bound index.
    if not interfaces and primary_ip:
        interfaces.append(Interface(None, primary_ip, None))

    # Default-route interface first -- it's where a LAN camera almost always is.
    interfaces.sort(key=lambda itf: itf.ip != primary_ip)
    return interfaces


def _build_probe() -> bytes:
    """A well-formed WS-Discovery Probe for NetworkVideoTransmitter devices.

    Note: ``mustUnderstand`` lives in the SOAP envelope namespace (``e:``); the
    original reference snippet referenced an undeclared ``a:`` prefix, which
    some strict stacks reject.
    """
    message_id = f"uuid:{uuid.uuid4()}"
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        "<e:Header>"
        f"<w:MessageID>{message_id}</w:MessageID>"
        '<w:To e:mustUnderstand="true">'
        "urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>"
        '<w:Action e:mustUnderstand="true">'
        "http://schemas.xmlsoap.org/ws:2005/04/discovery/Probe</w:Action>"
        "</e:Header>"
        "<e:Body>"
        "<d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe>"
        "</e:Body>"
        "</e:Envelope>"
    ).encode("utf-8")


def _parse_xaddrs(payload: str) -> list[tuple[str, int, str]]:
    results: list[tuple[str, int, str]] = []
    match = re.search(r"XAddrs>([^<]+)<", payload)
    if not match:
        return results
    for xaddr in match.group(1).split():
        url_match = re.search(r"https?://([^/:]+):?(\d*)", xaddr)
        if url_match:
            ip = url_match.group(1)
            port = int(url_match.group(2)) if url_match.group(2) else 80
            results.append((ip, port, xaddr))
    return results


def _probe_from(itf: Interface, timeout: float) -> list[tuple[str, int, str]]:
    """Send a Probe pinned to ``itf`` and drain replies until ``timeout``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Force egress below the routing table (macOS) so a VPN can't hijack it.
    if itf.index is not None:
        try:
            sock.setsockopt(socket.IPPROTO_IP, _IP_BOUND_IF, itf.index)
        except OSError:
            pass  # non-macOS / unsupported: fall back to IP_MULTICAST_IF below
    sock.setsockopt(
        socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(itf.ip)
    )
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout)

    found: list[tuple[str, int, str]] = []
    label = itf.name or itf.ip
    try:
        sock.bind((itf.ip, 0))
        sock.sendto(_build_probe(), (MULTICAST_GROUP, MULTICAST_PORT))
    except OSError as exc:
        print(f"  (probe via {label} failed: {exc})")
        if exc.errno == errno.EHOSTUNREACH:
            print(f"  -> {_LOCAL_NETWORK_HINT}")
        sock.close()
        return found

    try:
        while True:
            try:
                data, _addr = sock.recvfrom(65536)
            except socket.timeout:
                break
            payload = data.decode("utf-8", errors="ignore")
            found.extend(_parse_xaddrs(payload))
    finally:
        sock.close()
    return found


def discover_onvif_cameras(
    timeout: float = 3.0, bind_ip: str | None = None
) -> list[DiscoveredCamera]:
    """Broadcast a Probe on every candidate interface and collect responders."""
    interfaces = _candidate_interfaces(bind_ip)
    if not interfaces:
        print(
            "Could not determine a LAN interface to probe from. "
            "Set TAPO_BIND_IP to your Mac's LAN IP (e.g. 10.0.0.165)."
        )
        return []

    labels = ", ".join(f"{i.name or '?'}({i.ip})" for i in interfaces)
    print(
        f"Scanning the LAN for ONVIF devices via WS-Discovery "
        f"(waiting {timeout:.0f}s, interfaces: {labels})..."
    )

    discovered: list[DiscoveredCamera] = []
    seen_ips: set[str] = set()
    for itf in interfaces:
        for ip, port, xaddr in _probe_from(itf, timeout):
            if ip not in seen_ips:
                seen_ips.add(ip)
                discovered.append({"ip": ip, "port": port, "xaddr": xaddr})

    return discovered


if __name__ == "__main__":
    cams = discover_onvif_cameras()
    if not cams:
        print("No ONVIF cameras found.")
    for i, cam in enumerate(cams):
        print(f"  [{i}] {cam['ip']}:{cam['port']}  ({cam['xaddr']})")
