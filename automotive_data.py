"""
Automotive Companies Classification Data

Contains the classification data for automotive industry companies.
"""

AUTOMOTIVE_COMPANIES = {
    "Actia": "Autopartes electrónicas",
    "American Axle & Manufacturing": "Autopartes",
    "Amphenol": "Autopartes electrónicas", 
    "Aptiv": "Autopartes electrónicas",
    "Archermind": "My/oS electrónicos",
    "Arcosa": "",
    "Autoliv": "Autopartes",
    "Baidu": "My/oS electrónicos",
    "BMW Group": "OEM",
    "Borgwarner": "Autopartes electrónicas",
    "Brembo": "Autopartes",
    "BYD": "OEM",
    "CATL": "My/oS electrónicos",
    "CIE Automotive": "Autopartes",
    "Continental": "Autopartes",
    "Delta Electronics": "My/oS electrónicos",
    "Denso": "Autopartes electrónicas",
    "Eaton": "My/oS electrónicos",
    "FAW": "OEM",
    "Flextronics": "Autopartes electrónicas",
    "Ford Motor": "OEM",
    "FORVIA / HELLA / FAURECIA": "Autopartes electrónicas",
    "Fujikura": "My/oS electrónicos",
    "Geely": "OEM",
    "General Motors": "OEM",
    "Hitachi": "My/oS electrónicos",
    "Hon Hai": "My/oS electrónicos",
    "Honeywell International": "My/oS electrónicos",
    "HUAWEI": "My/oS electrónicos",
    "Integer Holding": "My/oS electrónicos",
    "Johnson Controls": "My/oS electrónicos",
    "Kia": "OEM",
    "Kyocera": "My/oS electrónicos",
    "Lear Corporation": "Autopartes electrónicas",
    "LG Electronics": "My/oS electrónicos",
    "Magna International": "Autopartes electrónicas",
    "Mahindra & Mahindra": "OEM",
    "Mitsubishi Electric": "My/oS electrónicos",
    "Nissan Motor": "OEM",
    "NXP Semiconductors": "Semiconductores",
    "ONSEMI": "Semiconductores",
    "OPmobility": "Autopartes electrónicas",
    "Panasonic Holdings": "My/oS electrónicos",
    "Pegatron": "My/oS electrónicos",
    "Qualcomm": "Semiconductores",
    "Quanta Computer": "My/oS electrónicos",
    "RENAULT": "OEM",
    "Renesas": "Semiconductores",
    "Robert Bosch": "My/oS electrónicos",
    "SAIC": "OEM",
    "Samsung Electronics": "My/oS electrónicos",
    "Sanmina SCI Systems": "My/oS electrónicos",
    "Schneider Electric": "My/oS electrónicos",
    "Sensata Tech": "My/oS electrónicos",
    "Sony": "My/oS electrónicos",
    "Stellantis": "OEM",
    "STMicroelectronics": "Semiconductores",
    "Tesla": "OEM",
    "Texas Instruments": "My/oS electrónicos",
    "Thales": "My/oS electrónicos",
    "Toyota Motor": "OEM",
    "Valeo": "Autopartes electrónicas",
    "Vinfast": "OEM",
    "Vishay": "Semiconductores",
    "Volkswagen": "OEM",
    "Volvo": "OEM",
    "XIAOMI": "OEM",
    "ZF": "Autopartes electrónicas"
}

# Color palette for different categories
INDUSTRY_COLORS = {
    "OEM": "#FF6B6B",                    # Red - Original Equipment Manufacturers
    "Autopartes": "#4ECDC4",             # Teal - Traditional Auto Parts
    "Autopartes electrónicas": "#45B7D1", # Blue - Electronic Auto Parts
    "My/oS electrónicos": "#96CEB4",     # Green - Electronic Systems & Software
    "Semiconductores": "#FECA57",        # Yellow - Semiconductors
    "": "#CCCCCC"                        # Gray - Uncategorized
}

def get_automotive_groups():
    """
    Get predefined automotive industry groups.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of group definitions ready for UniversalGroupManager
    """
    from datetime import datetime
    
    # Group companies by industry classification
    industry_groups = {}
    
    for company, industry in AUTOMOTIVE_COMPANIES.items():
        if not industry:  # Skip companies without classification
            continue
            
        if industry not in industry_groups:
            industry_groups[industry] = []
        industry_groups[industry].append(company)
    
    # Create group definitions
    groups = {}
    for industry, companies in industry_groups.items():
        group_name = f"Automotive_{industry.replace('/', '_').replace(' ', '_')}"
        
        groups[group_name] = {
            'name': group_name,
            'units': companies,
            'description': f"Automotive companies - {industry}",
            'color': INDUSTRY_COLORS.get(industry, "#CCCCCC"),
            'created': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'usage_count': 0
        }
    
    return groups

def load_automotive_sample_groups(group_manager):
    """
    Load automotive sample groups into the group manager.
    
    Args:
        group_manager: UniversalGroupManager instance
        
    Returns:
        bool: True if loaded successfully
    """
    try:
        automotive_groups = get_automotive_groups()
        
        loaded_count = 0
        for group_name, group_data in automotive_groups.items():
            # Check if group already exists
            if group_name not in group_manager.groups:
                group_manager.groups[group_name] = group_data
                loaded_count += 1
        
        if loaded_count > 0:
            group_manager.save_groups()
            group_manager.log_operation('load_automotive_sample', {
                'loaded_count': loaded_count,
                'total_companies': sum(len(g['units']) for g in automotive_groups.values())
            })
        
        return loaded_count > 0
        
    except Exception as e:
        print(f"Error loading automotive sample groups: {e}")
        return False

if __name__ == "__main__":
    # Print summary of automotive data
    print("Automotive Companies Classification Summary:")
    print("=" * 50)
    
    industry_counts = {}
    for company, industry in AUTOMOTIVE_COMPANIES.items():
        if industry:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
    
    for industry, count in sorted(industry_counts.items()):
        print(f"{industry}: {count} companies")
    
    print(f"\nTotal companies: {len(AUTOMOTIVE_COMPANIES)}")
    print(f"Categorized companies: {sum(industry_counts.values())}")
    print(f"Uncategorized companies: {len(AUTOMOTIVE_COMPANIES) - sum(industry_counts.values())}")