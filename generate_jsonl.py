import json

arxiv_data = [
    {"id": "0704.0001", "submitter": "Pavel Nadolsky", "authors": "Bal'azs, C., Berger, E. L., Nadolsky, P. M., Yuan, C.-P.", "title": "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies", "categories": "hep-ph", "abstract": "A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders...", "update_date": "2008-11-26", "authors_parsed": [["Bal", "zs", ""], ["Berger", "E. L.", ""], ["Nadolsky", "P. M.", ""], ["Yuan", "C.-P.", ""]]},
    {"id": "0704.0002", "submitter": "Louis Theran", "authors": "Streinu, Ilyas, Theran, Louis", "title": "Sparsity-certifying Graph Decompositions", "categories": "math.CO cs.CG", "abstract": "We describe a new algorithm, the (k,l)-pebble game with colors...", "update_date": "2008-12-13", "authors_parsed": [["Streinu", "Ilyas", ""], ["Theran", "Louis", ""]]},
    {"id": "0704.0003", "submitter": "Test Abstract", "authors": "A, B.", "title": "No abstract provided", "categories": "math.CO cs.CG", "abstract": None, "update_date": "2009-01-01", "authors_parsed": [["A", "B", ""]]}
]

s2orc_data = [
    {"paper_id": "123", "title": "A Machine Learning Approach to Quantum Computing", "authors": [{"first": "Alice", "last": "Smith"}, {"first": "Bob", "last": "Jones"}], "abstract": "We present a novel machine learning approach to optimize quantum circuits...", "year": 2023, "venue": "Nature"},
    {"paper_id": "124", "title": "Null Abstract Example", "authors": [{"first": "Charlie", "last": "Brown"}], "abstract": None, "year": 2022, "venue": "Science"}
]

with open('sample_arxiv.json', 'w') as f:
    for item in arxiv_data:
        f.write(json.dumps(item) + '\n')

with open('sample_s2orc.json', 'w') as f:
    for item in s2orc_data:
        f.write(json.dumps(item) + '\n')

