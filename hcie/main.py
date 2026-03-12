from hcie.database_search import DatabaseSearch

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "smiles",
        action="store",
        type=str,
        help="<Required> smiles string of molecule to search, with attachment points identified",
    )
    parser.add_argument(
        "-n", "--name", action="store", type=str, help="name of molecule"
    )
    parser.add_argument(
        "--output-dir",
        action="store",
        type=str,
        help="Optional output directory for results",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing output files (CSV/SDF/PNG)",
    )
    parser.add_argument(
        "--return-rdkit-mols",
        action="store_true",
        help="Return RDKit Mol objects (prints count in CLI)",
    )

    return parser.parse_args()


def main():
    args = get_args()
    smiles = args.smiles
    name = args.name
    search = DatabaseSearch(
        smiles,
        name=name,
        output_dir=args.output_dir,
        write_files=not args.no_write,
        return_rdkit_mols=args.return_rdkit_mols,
    )
    rdkit_mols = search.search()
    if args.return_rdkit_mols:
        rdkit_mols = rdkit_mols or []
        print(f"Returned {len(rdkit_mols)} RDKit molecules.")


if __name__ == "__main__":
    main()
