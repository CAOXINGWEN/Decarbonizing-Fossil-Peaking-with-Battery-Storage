"""Convert the three native-EUR fields in the v73 paired ledger to USD.

Only the declared fields are rewritten. Other CSV cell strings are preserved.
The identities reject an already converted input and any inconsistent ledger.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse, csv, hashlib, json

FIELDS=('accounting_difference','incremental_efc_credit','avoided_residual_peak_credit')
RATE=Decimal('1.1300')
TOL=Decimal('0.00000001')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def convert(source, destination, audit_path=None):
    with source.open(newline='') as f:
        reader=csv.DictReader(f); names=reader.fieldnames; rows=list(reader)
    assert rows and all(c in names for c in FIELDS)
    before=[row.copy() for row in rows]
    def dec(r,c): return Decimal(r[c])
    with localcontext() as ctx:
        ctx.prec=40
        raw_error=max(abs(dec(r,'efc_value')-dec(r,'system_cost_value')-dec(r,'accounting_difference')) for r in rows)
        assert raw_error>TOL, 'Input is already in USD or requires explicit unit review; no conversion performed.'
        for r in rows:
            difference=dec(r,'efc_value')-dec(r,'system_cost_value')
            assert abs(difference-dec(r,'accounting_difference_from_capacity_terms'))<TOL
            assert abs(difference-RATE*dec(r,'accounting_difference'))<TOL
            assert abs(dec(r,'incremental_efc_credit')-dec(r,'avoided_residual_peak_credit')-dec(r,'accounting_difference'))<TOL
            for c in FIELDS: r[c]=format(dec(r,c)*RATE,'f')
        errors={
            'valuation_difference':max(abs(dec(r,'efc_value')-dec(r,'system_cost_value')-dec(r,'accounting_difference')) for r in rows),
            'capacity_component_difference':max(abs(dec(r,'incremental_efc_credit')-dec(r,'avoided_residual_peak_credit')-dec(r,'accounting_difference')) for r in rows),
            'derived_capacity_difference':max(abs(dec(r,'incremental_efc_credit')-dec(r,'avoided_residual_peak_credit')-dec(r,'accounting_difference_from_capacity_terms')) for r in rows)}
        assert max(errors.values())<TOL
    for a,b in zip(before,rows):
        assert all(a[c]==b[c] for c in names if c not in FIELDS)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    audit={'input_sha256':sha(source),'output_sha256':sha(destination),'rows':len(rows),'scaled_columns':list(FIELDS),'usd_per_eur':str(RATE),'other_cell_strings_unchanged':True,'identities_max_absolute_error_usd_kw_year':{k:float(v) for k,v in errors.items()},'passed':True}
    if audit_path:audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('--audit',type=Path)
    a=p.parse_args();print(json.dumps(convert(a.source,a.destination,a.audit),indent=2))
