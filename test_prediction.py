#!/usr/bin/env python3
"""
Quick test script to verify predict.py works with saved models
"""

import json
import subprocess
import sys

# Sample features (simplified for testing)
test_features = {
    "IsBetaUser": 0,
    "RealTimeProtectionState": 7,
    "IsPassiveModeEnabled": 0,
    "AntivirusConfigID": 5840,
    "NumAntivirusProductsInstalled": 1,
    "NumAntivirusProductsEnabled": 1,
    "CityID": 67785,
    "OrganizationID": 0,
    "GeoNameID": 203,
    "IsSystemProtected": 1,
    "SMode": 0,
    "IEVersionID": 11,
    "FirewallEnabled": 1,
    "EnableLUA": 1,
    "Census_OEMNameID": 2827,
    "Census_OEMModelID": 5929,
    "Census_ProcessorCoreCount": 4,
    "Census_ProcessorManufacturerID": 2,
    "Census_ProcessorModelID": 1045,
    "Census_PrimaryDiskCapacityMB": 476940,
    "Census_PrimaryDiskType": 0,
    "Census_SystemVolumeCapacityMB": 476940,
    "Census_TotalPhysicalRAMMB": 8192,
    "Census_ChassisType": 3,
    "Census_PrimaryDisplayDiagonalInches": 15.6,
    "Census_PrimaryDisplayResolutionHorizontal": 1920,
    "Census_PrimaryDisplayResolutionVertical": 1080,
    "Census_InternalBatteryNumberOfCharges": 0,
    "Census_OSInstallLanguageID": 1033,
    "Census_IsFlightsDisabled": 0,
    "Census_FirmwareManufacturerID": 1,
    "Census_FirmwareVersionID": 1,
    "Census_IsVirtualDevice": 0,
    "Census_IsAlwaysOnAlwaysConnectedCapable": 0,
    "IsGamer": 1,
    "RegionIdentifier": 10,
    "MachineID": "test123",
    "ProductName": "win8defender",
    "EngineVersion": "1.1.15200.1",
    "AppVersion": "4.18.1807.18075",
    "SignatureVersion": "1.273.1420.0",
    "Platform": "windows10",
    "Processor": "x64",
    "OSVersion": "10.0.17134.1",
    "OSBuildNumber": 17134,
    "OSBuildNumberOnly": 17134,
    "OSBuildLab": "rs4_release.180410-1804",
    "OSPlatformSubRelease": "rs4",
    "OSVer": "10.0",
    "SkuEdition": "Professional",
    "PuaMode": "0",
    "SmartScreen": "on",
    "Census_ProcessorClass": "X86",
    "Census_PrimaryDiskTypeName": "HDD",
    "Census_HasOpticalDiskDrive": 0,
    "Census_OSVersion": "10.0.17134.1",
    "Census_OSArchitecture": "amd64",
    "Census_OSBranch": "rs4_release",
    "Census_OSEdition": "Professional",
    "Census_OSSkuName": "PROFESSIONAL",
    "Census_OSInstallTypeName": "Client",
    "Census_OSWUAutoUpdateOptionsName": "Auto",
    "Census_GenuineStateName": "GENUINE",
    "Census_ActivationChannel": "Retail",
    "Census_FlightRing": "Disabled",
    "Census_ThresholdOptIn": 0,
    "Census_IsSecureBootEnabled": 1,
    "Census_IsWIMBootEnabled": 0,
    "Census_IsTouchEnabled": 0,
    "Census_IsPenCapable": 0,
    "Census_DeviceFamily": "Windows.Desktop",
    "Census_DeviceModel": "Laptop",
    "DateOS": 180410
}

print("Testing predict.py with saved ML models...")
print("=" * 60)

try:
    # Call predict.py with test features
    process = subprocess.Popen(
        ['./venv/bin/python', 'predict.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send features via stdin
    input_data = json.dumps({'features': test_features})
    stdout, stderr = process.communicate(input=input_data, timeout=30)
    
    if process.returncode == 0:
        print("✓ predict.py executed successfully!")
        print("\nPredictions:")
        result = json.loads(stdout)
        
        print("\nML Models:")
        for model_name, pred in result.get('ml_models', {}).items():
            print(f"  {model_name}: prediction={pred['prediction']}, confidence={pred['confidence']:.2%}")
        
        print("\nDL Models:")
        if result.get('dl_models'):
            for model_name, pred in result.get('dl_models', {}).items():
                print(f"  {model_name}: prediction={pred['prediction']}, confidence={pred['confidence']:.2%}")
        else:
            print("  (DL models not yet trained)")
        
        print("\n✓ All systems working!")
    else:
        print("✗ Error running predict.py")
        print("\nStderr:", stderr)
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print("✗ Timeout - predict.py took too long")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
