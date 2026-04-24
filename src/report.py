VIOLATIONS = {"cigarette","lighter","match","solvent","spray_gun","two_pack_paint","rust_remover"}
COMPLIANCE = {"mask","gloves","goggles","compressor","gauge"}

def generate_report(detections):
    hazards = [d for d in detections if d in VIOLATIONS]
    safe = [d for d in detections if d in COMPLIANCE]

    risk = "LOW"
    if len(hazards) > 2:
        risk = "HIGH"
    elif hazards:
        risk = "MEDIUM"

    return f"Hazards: {hazards}\nSafe: {safe}\nRisk: {risk}"
