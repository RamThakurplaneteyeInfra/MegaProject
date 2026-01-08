from lxml import etree
import xml.etree.ElementTree as ET
def parse_kml(content: bytes):
    root = etree.fromstring(content)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    placemarks = root.findall(".//kml:Placemark", ns)
    villages = []

    for pm in placemarks:
        name = pm.findtext("kml:name", namespaces=ns)

        schema = pm.find(".//kml:SchemaData", ns)
        data = {sd.get("name"): sd.text for sd in schema.findall("kml:SimpleData", ns)}

        coords_text = pm.findtext(
            ".//kml:coordinates", namespaces=ns
        ).strip()

        coords = [
            [float(x), float(y)]
            for x, y, *_ in (c.split(",") for c in coords_text.split())
        ]

        villages.append({
            "village_name": name,
            "district": data.get("DISTRICT"),
            "sub_dist": data.get("SUB_DIST"),
            "state": data.get("STATE"),
            "coordinates": coords
        })

    return villages
def parse_district_kml(kml_bytes: bytes):
    ns = {
        "kml": "http://www.opengis.net/kml/2.2",
        "ns0": "http://www.opengis.net/kml/2.2"
    }

    root = ET.fromstring(kml_bytes)
    placemarks = root.findall(".//ns0:Placemark", ns)

    districts = []

    for pm in placemarks:
        # District name
        dist_name = None
        for sd in pm.findall(".//ns0:SimpleData", ns):
            if sd.attrib.get("name") == "Dist_Name":
                dist_name = sd.text.strip()

        if not dist_name:
            continue

        # Coordinates
        coord_text = pm.find(
            ".//ns0:coordinates", ns
        ).text.strip()

        coords = []
        for c in coord_text.split():
            lon, lat = map(float, c.split(",")[:2])
            coords.append((lon, lat))

        districts.append({
            "district": dist_name,
            "state": "Maharashtra",
            "coordinates": coords
        })

    return districts