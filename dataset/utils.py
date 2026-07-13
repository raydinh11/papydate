def _parse_year(date_str, fallback):
    s = str(date_str).strip()
    if s in ("nan", "", "None"):
        return fallback
    try:
        if s.startswith("-"):
            return -int(s[1:].split("-")[0])
        return int(s.split("-")[0])
    except (ValueError, IndexError):
        return fallback
