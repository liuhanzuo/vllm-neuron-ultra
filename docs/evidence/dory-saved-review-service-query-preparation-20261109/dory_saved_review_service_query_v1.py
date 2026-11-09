"""Source-only candidate for the saved review's repeated service query.

Every check makes a fresh synchronous ListUnitsFiltered request on one private
connection. No result cache, polling interval, service mutation or admission.
The running reviewer is unchanged; a future executor needs a separate source
and dependency review and the actual terminal history before using this code.
"""
import os
from pathlib import Path
import re
import stat

ADDRESS = 'unix:path=/run/user/1000/bus'
STATES = ('running', 'activating', 'deactivating')
SIGNATURE = '(ssssssouso)'
MAX_ROWS = 4096
MAX_TEXT_BYTES = 256 * 1024
OBJECT_PATH = re.compile(r'/(?:[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*)?\Z')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def service_names(rows, signature):
    """Apply the original service type, state and trn1 prefix selection."""
    require(signature == SIGNATURE, 'Exact systemd ListUnitsFiltered signature')
    require(isinstance(rows, (list, tuple)) and len(rows) <= MAX_ROWS,
            'Bounded complete service response')
    names, seen, total = [], set(), 0
    for row in rows:
        require(isinstance(row, (list, tuple)) and len(row) == 10,
                'Complete original systemd unit row')
        strings = [row[i] for i in range(10) if i != 7]
        require(all(isinstance(x, str) for x in strings), 'Typed unit strings')
        require(isinstance(row[7], int) and not isinstance(row[7], bool)
                and 0 <= row[7] <= 2**32 - 1, 'Typed uint32 unit job ID')
        total += sum(len(x.encode('utf-8')) for x in strings)
        require(total <= MAX_TEXT_BYTES, 'Bounded decoded service text')
        name = str(row[0])
        require(name and not any(c.isspace() or ord(c) < 32 for c in name)
                and name not in seen, 'Unique complete unit name')
        seen.add(name)
        require(OBJECT_PATH.fullmatch(row[6]) and OBJECT_PATH.fullmatch(row[9]),
                'Typed unit and job object paths')
        require(row[3] in STATES or row[4] in STATES,
                'Server response satisfies the exact original state filter')
        if name.endswith('.service') and name.startswith('trn1-'):
            names.append(name.removesuffix('.service'))
    return names


class PersistentServiceQuery:
    def __init__(self, request, close):
        self._request, self._close = request, close
        self.poisoned = self.closed = False
        self.requests = 0

    def check(self, excluded=()):
        require(not self.closed and not self.poisoned,
                'Closed or failed service query cannot resume')
        try:
            require(isinstance(excluded, (list, tuple))
                    and all(isinstance(x, str) and x.startswith('trn1-')
                            and not x.endswith('.service') for x in excluded),
                    'Explicit original unit exclusions')
            self.requests += 1
            rows, signature = self._request(STATES)
            names = service_names(rows, signature)
            require(not [name for name in names if name not in excluded],
                    'Competing active heavy service')
            return names
        except BaseException:
            self.poisoned = True
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self._close()


def connect_fixed_user_bus():
    """Use the same UID, address, private directory and socket requirements."""
    require(os.getuid() == os.geteuid() == 1000, 'UID1000 required')
    require(os.environ.get('XDG_RUNTIME_DIR') == '/run/user/1000'
            and os.environ.get('DBUS_SESSION_BUS_ADDRESS') == ADDRESS,
            'Exact original user bus environment')
    runtime = Path('/run/user/1000')
    directory, socket = runtime.lstat(), (runtime / 'bus').lstat()
    require(stat.S_ISDIR(directory.st_mode) and stat.S_IMODE(directory.st_mode) == 0o700
            and directory.st_uid == 1000 and runtime.resolve() == runtime,
            'Original private user runtime directory')
    require(stat.S_ISSOCK(socket.st_mode) and socket.st_uid == 1000,
            'Original user bus socket')
    import dbus
    bus = dbus.bus.BusConnection(ADDRESS)
    try:
        require(bus.get_is_connected(), 'Connected private user bus')

        def request(states):
            require(bus.get_is_connected(), 'User bus connection lost')
            rows = bus.call_blocking(
                'org.freedesktop.systemd1', '/org/freedesktop/systemd1',
                'org.freedesktop.systemd1.Manager', 'ListUnitsFiltered', 'as',
                (dbus.Array(states, signature='s'),), timeout=15)
            return rows, str(rows.signature)

        return PersistentServiceQuery(request, bus.close)
    except BaseException:
        bus.close()
        raise


if __name__ == '__main__':
    raise SystemExit('Source preparation only; no executor or runtime admission')
