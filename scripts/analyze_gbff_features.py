#!/usr/bin/env python3
"""
Quick script to analyze features in GenBank gbff files

This script reads a GenBank file and analyzes what types of features
are available in the records.
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def analyze_gbff_features(file_path: str, max_records: int = 10, max_length: int = 1024) -> Dict:
    """
    Analyze features in a GenBank file, focusing on biological features in sequences < max_length.
    
    Args:
        file_path: Path to the GenBank file
        max_records: Maximum number of records to analyze
        max_length: Maximum sequence length to consider
        
    Returns:
        Dictionary with analysis results
    """
    print(f"🔍 Analyzing GenBank file: {file_path}")
    print(f"📊 Analyzing up to {max_records} records with length < {max_length}...")
    print("=" * 60)
    
    # Statistics containers
    feature_types = Counter()
    qualifier_keys = defaultdict(Counter)
    molecule_types = Counter()
    organisms = Counter()
    record_lengths = []
    
    # Biological features tracking (like RefSeqPreprocessor)
    biological_features = {
        'signal_peptides': Counter(),
        'transmembrane_regions': Counter(),
        'disulfide_bonds': Counter(),
        'glycosylation_sites': Counter(),
        'phosphorylation_sites': Counter()
    }
    
    # Records with biological features
    records_with_bio_features = []
    
    # Sample records for detailed inspection
    sample_records = []
    
    try:
        with open(file_path) as handle:
            for i, record in enumerate(SeqIO.parse(handle, "genbank")):
                if i >= max_records:
                    break
                    
                # Check sequence length first
                seq_length = len(record.seq)
                if seq_length >= max_length:
                    print(f"\n⏭️  Skipping Record {i+1}: {record.id} (length {seq_length} >= {max_length})")
                    continue
                
                print(f"\n📋 Record {i+1}: {record.id}")
                print(f"   Description: {record.description}")
                print(f"   Length: {seq_length}")
                print(f"   Organism: {record.annotations.get('organism', 'N/A')}")
                print(f"   Molecule type: {record.annotations.get('molecule_type', 'N/A')}")
                
                # Count molecule types and organisms
                molecule_types[record.annotations.get('molecule_type', 'unknown')] += 1
                organisms[record.annotations.get('organism', 'unknown')] += 1
                record_lengths.append(seq_length)
                
                # Track biological features (like RefSeqPreprocessor)
                has_bio_features = False
                bio_features_in_record = []
                
                # Analyze features
                print(f"   Features ({len(record.features)} total):")
                for j, feature in enumerate(record.features):
                    feature_type = feature.type
                    feature_types[feature_type] += 1
                    
                    print(f"     {j+1}. {feature_type} - {feature.location}")
                    
                    # Check for biological features (like RefSeqPreprocessor)
                    if feature_type == "sig_peptide":
                        biological_features['signal_peptides'][record.id] += 1
                        has_bio_features = True
                        bio_features_in_record.append("signal_peptide")
                        print(f"        🧬 BIOLOGICAL FEATURE: signal_peptide")
                    elif feature_type == "transmembrane":
                        biological_features['transmembrane_regions'][record.id] += 1
                        has_bio_features = True
                        bio_features_in_record.append("transmembrane")
                        print(f"        🧬 BIOLOGICAL FEATURE: transmembrane")
                    elif feature_type == "disulfide":
                        biological_features['disulfide_bonds'][record.id] += 1
                        has_bio_features = True
                        bio_features_in_record.append("disulfide_bond")
                        print(f"        🧬 BIOLOGICAL FEATURE: disulfide_bond")
                    elif feature_type == "glycosylation":
                        biological_features['glycosylation_sites'][record.id] += 1
                        has_bio_features = True
                        bio_features_in_record.append("glycosylation")
                        print(f"        🧬 BIOLOGICAL FEATURE: glycosylation")
                    elif feature_type == "phosphorylation":
                        biological_features['phosphorylation_sites'][record.id] += 1
                        has_bio_features = True
                        bio_features_in_record.append("phosphorylation")
                        print(f"        🧬 BIOLOGICAL FEATURE: phosphorylation")
                    
                    # Analyze qualifiers
                    for key, values in feature.qualifiers.items():
                        qualifier_keys[feature_type][key] += 1
                        if len(values) == 1:
                            print(f"        {key}: {values[0]}")
                        else:
                            print(f"        {key}: {len(values)} values")
                
                # Track records with biological features
                if has_bio_features:
                    records_with_bio_features.append({
                        'id': record.id,
                        'length': seq_length,
                        'organism': record.annotations.get('organism', 'unknown'),
                        'features': bio_features_in_record
                    })
                    print(f"   🧬 Found biological features: {', '.join(bio_features_in_record)}")
                
                # Store sample for detailed analysis
                if i < 3:  # Keep first 3 records for detailed analysis
                    sample_records.append(record)
                    
                print("-" * 40)
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return {}
    
    # Compile results
    results = {
        'file_path': file_path,
        'records_analyzed': min(max_records, i + 1),
        'max_length_filter': max_length,
        'feature_types': feature_types,  # Keep as Counter
        'qualifier_keys': {ft: qk for ft, qk in qualifier_keys.items()},  # Keep as Counter
        'molecule_types': molecule_types,  # Keep as Counter
        'organisms': organisms,  # Keep as Counter
        'biological_features': biological_features,  # New: biological features tracking
        'records_with_bio_features': records_with_bio_features,  # New: records with bio features
        'length_stats': {
            'min': min(record_lengths) if record_lengths else 0,
            'max': max(record_lengths) if record_lengths else 0,
            'mean': sum(record_lengths) / len(record_lengths) if record_lengths else 0,
            'count': len(record_lengths)
        },
        'sample_records': sample_records
    }
    
    return results


def print_analysis_summary(results: Dict):
    """Print a summary of the analysis results."""
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n📁 File: {results['file_path']}")
    print(f"📋 Records analyzed: {results['records_analyzed']}")
    print(f"🔍 Max length filter: {results['max_length_filter']}")
    
    print(f"\n📏 Sequence Length Statistics:")
    stats = results['length_stats']
    print(f"   Min: {stats['min']:,}")
    print(f"   Max: {stats['max']:,}")
    print(f"   Mean: {stats['mean']:,.1f}")
    print(f"   Count: {stats['count']}")
    
    print(f"\n🧬 Biological Features Summary:")
    bio_features = results['biological_features']
    total_records_with_bio = len(results['records_with_bio_features'])
    print(f"   Records with biological features: {total_records_with_bio}")
    
    for feature_type, counter in bio_features.items():
        count = len(counter)
        if count > 0:
            print(f"   {feature_type}: {count} records")
            # Show first few record IDs
            record_ids = list(counter.keys())[:3]
            print(f"     Examples: {', '.join(record_ids)}")
            if len(counter) > 3:
                print(f"     ... and {len(counter) - 3} more")
    
    if total_records_with_bio > 0:
        print(f"\n🧬 Records with Biological Features:")
        for record in results['records_with_bio_features'][:5]:  # Show first 5
            print(f"   {record['id']} (length: {record['length']}) - {', '.join(record['features'])}")
        if len(results['records_with_bio_features']) > 5:
            print(f"   ... and {len(results['records_with_bio_features']) - 5} more")
    
    print(f"\n🧬 Molecule Types:")
    molecule_types = results['molecule_types']
    if hasattr(molecule_types, 'most_common'):
        for mol_type, count in molecule_types.most_common():
            print(f"   {mol_type}: {count}")
    else:
        # Fallback for dict
        for mol_type, count in sorted(molecule_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {mol_type}: {count}")
    
    print(f"\n🦠 Top Organisms:")
    organisms = results['organisms']
    if hasattr(organisms, 'most_common'):
        for organism, count in organisms.most_common(5):
            print(f"   {organism}: {count}")
    else:
        # Fallback for dict
        for organism, count in sorted(organisms.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {organism}: {count}")
    
    print(f"\n🏷️  Feature Types:")
    feature_types = results['feature_types']
    if hasattr(feature_types, 'most_common'):
        for feature_type, count in feature_types.most_common():
            print(f"   {feature_type}: {count}")
    else:
        # Fallback for dict
        for feature_type, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {feature_type}: {count}")
    
    print(f"\n🔑 Common Qualifier Keys by Feature Type:")
    for feature_type, qualifiers in results['qualifier_keys'].items():
        if qualifiers:
            print(f"   {feature_type}:")
            if hasattr(qualifiers, 'most_common'):
                for key, count in qualifiers.most_common(5):
                    print(f"     {key}: {count}")
            else:
                # Fallback for dict
                for key, count in sorted(qualifiers.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     {key}: {count}")


def analyze_sample_record_detailed(record, record_num: int):
    """Perform detailed analysis of a sample record."""
    print(f"\n🔬 DETAILED ANALYSIS - Record {record_num}")
    print("=" * 60)
    
    print(f"ID: {record.id}")
    print(f"Name: {record.name}")
    print(f"Description: {record.description}")
    print(f"Length: {len(record.seq):,}")
    print(f"Organism: {record.annotations.get('organism', 'N/A')}")
    print(f"Molecule type: {record.annotations.get('molecule_type', 'N/A')}")
    
    # Show all annotations
    print(f"\n📝 All Annotations:")
    for key, value in record.annotations.items():
        if isinstance(value, list):
            print(f"   {key}: {len(value)} items")
            if len(value) <= 3:
                for item in value:
                    print(f"     - {item}")
        else:
            print(f"   {key}: {value}")
    
    # Analyze features in detail
    print(f"\n🧬 Features Analysis:")
    for i, feature in enumerate(record.features):
        print(f"\n   Feature {i+1}: {feature.type}")
        print(f"   Location: {feature.location}")
        print(f"   Strand: {feature.location.strand}")
        
        if feature.qualifiers:
            print(f"   Qualifiers:")
            for key, values in feature.qualifiers.items():
                if len(values) == 1:
                    print(f"     {key}: {values[0]}")
                else:
                    print(f"     {key}: {len(values)} values")
                    for j, value in enumerate(values[:3]):  # Show first 3
                        print(f"       {j+1}. {value}")
                    if len(values) > 3:
                        print(f"       ... and {len(values) - 3} more")


def main():
    """Main function to run the analysis."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_gbff_features.py <gbff_file> [max_records] [max_length]")
        print("Example: python analyze_gbff_features.py ../data/vertebrate_mammalian.1.rna.gbff 5 1024")
        sys.exit(1)
    
    file_path = sys.argv[1]
    max_records = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    # Run analysis
    results = analyze_gbff_features(file_path, max_records, max_length)
    
    if not results:
        print("❌ Analysis failed")
        sys.exit(1)
    
    # Print summary
    print_analysis_summary(results)
    
    # Detailed analysis of sample records
    if results['sample_records']:
        print(f"\n🔬 DETAILED SAMPLE ANALYSIS")
        print("=" * 60)
        
        for i, record in enumerate(results['sample_records']):
            analyze_sample_record_detailed(record, i + 1)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Found {len(results['feature_types'])} different feature types")
    print(f"🦠 Found {len(results['organisms'])} different organisms")
    print(f"🧬 Found {len(results['records_with_bio_features'])} records with biological features")


if __name__ == "__main__":
    main() 