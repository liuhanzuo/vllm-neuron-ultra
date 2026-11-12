# MiniCPM5 SDK Python3.10 signed-origin preparation gap

The exact SDK symlink chain is /home/ubuntu/venvs/vllm_neuron/bin/python -> python3.10 -> /usr/bin/python3.10. Current dpkg text records version3.10.12-1~22.04.18, amd64. Only lstat/symlink observations were made on native files, never content reads.

| Required object | Exact package/version | Current bytes (lstat) | Provenance still missing |
|---|---|---:|---|
| /usr/bin/python3.10 | python3.10-minimal 3.10.12-1~22.04.18 |5,941,864| Exact signed archive/member replay, actual full hash and ELF |
| /usr/lib/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so | libpython3.10-stdlib 3.10.12-1~22.04.18 |132,120| Exact signed archive/member replay, full hash, DT_NEEDED and actual provider selection |
| /usr/lib/python3.10/ctypes/__init__.py | same libpython3.10-stdlib |17,988| Exact signed member SHA256 and authenticated execution origin |
| /usr/lib/python3.10/ctypes/_endian.py | same libpython3.10-stdlib |2,000| Exact signed member SHA256 and authenticated execution origin |
| /usr/lib/x86_64-linux-gnu/libffi.so.8.1.0 | libffi8 3.4.2-4 |47,688| Exact signed archive/member replay, full hash, positive FFI definitions and mapped provider |

/usr/lib/x86_64-linux-gnu/libffi.so.8 is a symlink to libffi.so.8.1.0. The canonical terminal must be authenticated; the SONAME-looking symlink alone does not prove selected FFI provider.

## Existing evidence that can be reused

All three current uncompressed APT Packages texts match the byte count and SHA256 in the already accepted signed Ubuntu Release chain. The saved Release texts themselves were rehashed, their SHA256 stanza entries were matched, and the saved successful signature status references were rehashed. No new gpgv execution or binary/keyring read occurred.

- jammy main:12ce0797a6ed39a1fee1321985976c6b78872a8fe4c7ef2fb712f2364c9839a3,6,779,186B.
- jammy-updates main:538dba84cd58fa414dc7242ba7d6cb4b58d42535a2b8d8ef3ba5c3d79f73c494,22,511,228B.
- jammy-security main:ff06c59ca6bd70d1e236457656292afc51a97c1e17fe5916f3043a02b0091510,21,028,804B.

This reuses full signed-release ancestry for exact package addresses, not Python3.12 ABI evidence. The original signed result is /mnt/nvme/dory-serial-inventory-ubuntu-metadata-r5-20261111/signed-metadata/sources-result.json, SHA59ae44799afcb4504177f2627a78177c783b40c671749a909fde3cf7b8b06409.

| Archive | Bytes | SHA256 |
|---|---:|---|
| python3.10-minimal_3.10.12-1~22.04.18_amd64.deb |2,277,402|34e4c7383fdf2562d78321acfee5aa1f7b92be371dd79410496ad35cb773f326|
| libpython3.10-stdlib_3.10.12-1~22.04.18_amd64.deb |1,850,286|ee09afbc981a4aaa00053f019eab6b45dcc4dbeb4ca4fcc4277eff539e09f228|
| libffi8_3.4.2-4_amd64.deb |22,326|b4f88c91fa6f4c942097be6abfc365fb133c5e147640168cbb7704fd855eac10|

These three archives total4,150,014B. The JSON report contains exact Ubuntu/security URLs and complete index references. No accepted complete member replay was found for these three packages in the inspected saved provenance. Local dpkg MD5/list/status records are identification hints, not independent current-content authentication.

The existing six-package replay at /mnt/nvme/dory-serial-inventory-deb-replay-r1-20261111/packages/sources-result.json (SHA594f9c3fa05579fa1da9af0a486ca252d988ff16e05041bd05b1b18237574a14) provides signed archive/member ancestry for12 canonical system files. All12 current lstat identities still match their saved observations, and installed package versions match. The seven libc6 members are version2.35-0ubuntu3.15; /usr/lib/x86_64-linux-gnu/libc.so.6 and ld-linux-x86-64.so.2 are explicit SDK prerequisites. The remaining libc members and libgcc/libgomp/libstdc++/libnuma/tcmalloc origins are reusable only if the actual SDK closure selects those exact files. Current full hashes and actual loader selection are still pending; same lstat identity is not a new full hash.

## Remaining requirements and minimum next serial work

The actual SDKNativeAuthenticator requires a separate accepted minicpm5_sdk310_native_source_review_v1 decision, complete signed Release -> package archive -> selected member records for every native/pure source, actual executable identity, no loader environment interposition, complete native hashes and ELF dependency providers, six positive FFI function definitions, and exact actual pre/post-ctypes mappings. It requires the post set to equal all declared native files. Python3.12 archive/built-in _ctypes/embedded FFI evidence cannot satisfy this separate SDK310 authority.

The known three packages are only the first missing package group, not a complete ELF closure. Bootstrap source also imports hashlib and json: package lists locate _hashlib in libpython3.10-minimal and _json in libpython3.10-stdlib. Their actual mapping state was not observed. Package Depends also names potential libexpat, zlib and OpenSSL providers. The report includes exact signed addresses for libpython3.10-minimal3.10.12-1~22.04.18, libssl3 3.0.2-0ubuntu1.30, libexpat1 2.4.7-1ubuntu0.9, zlib1g1:1.2.11.dfsg-2ubuntu9.2 and libcrypt1 1:4.4.27-1 as candidates. Package Depends is not substituted for DT_NEEDED or actual maps.

The smallest useful next serial operation is root-reviewed exact archive/member authentication for the missing primary package group, plus complete original SDK310 ELF/provider inventory to discover any additional providers before admission. Reuse the already verified Release/index chain and exact same-path system-member origins; do not install or alter the SDK environment. Any extra providers need the same signed member chain, then a root-reviewed complete decision. Only afterwards can the existing worker perform its full hash/ELF/FFI/pre-post mapping checks and original SDK import diagnostic. This report does not launch any of these operations or modify a live handoff.

Original9 native checks,1,284 scientific descriptor rows, AS2GiB, CPU600/605seconds, FSIZE2MiB, original inherited900seconds and serial/drain rules remain unchanged. The previous saved failure is libcusparseLt.so.0: failed to map segment from shared object. Its saved AS was2GiB and peak virtual size1,699,632KiB. Address-space pressure is a possible explanation, but there is no captured errno or complete rejected-mapping/ELF evidence. It is not a determined cause or a basis here to enlarge limits.

Validation: the source/metadata analysis command completed8f9e82/exit0, checking exact index-to-Release hashes, saved signature status references, six-package archive/member JSON links, installed versions and12 current lstat identities. No native binary, ELF, .deb, model/capture payload or weights were read; no ctypes/native import, download, install, lock/service operation or actual process observation occurred.
