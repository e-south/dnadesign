"""
regulondb_client.py

A Python module for interacting with a local RegulonDB GraphQL instance
and parsing TF-binding data from CSV files.
"""
import requests
import json

# The GraphQL endpoint of your local RegulonDB instance
endpoint = "http://localhost:7001/graphql"  # Replace 7001 with your actual port if different

# The GraphQL query to retrieve data
query = """
{
    getTFBindingById(_id:"ECOLIBS000000001")
    {
        _id
        closestGenes {
            _id
            name
            distanceTo
        }
        datasetIds
    }
}
"""

# Send the query to the GraphQL endpoint
r = requests.post(endpoint, json={"query": query})

# Check the response
if r.status_code == 200:
    try:
        # Pretty-print the JSON response
        print(json.dumps(r.json(), indent=2))
    except json.JSONDecodeError:
        print("The response could not be parsed as JSON.")
        print(f"Response Text: {r.text}")
else:
    raise Exception(f"Query failed to run with a {r.status_code}. Response: {r.text}")






# Handle possible errors
if "errors" in resp_data:
    print("GraphQL returned errors:", resp_data["errors"])
    exit(1)

all_datasets = resp_data["data"]["getDataSetList"]["data"]

# Prepare CSV output
with open("chipseq_binding_sites.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # You can define whatever columns you want; here is an example:
    writer.writerow(["datasetId", "TF_name", "targetGene", "bindingSiteSequence"])

    # 2) For each dataset, fetch its TFBS data
    for ds in all_datasets:
        ds_id = ds["datasetId"]
        tf_name = ds["factorName"]
        
        # GraphQL query to get all TFBS for this dataset
        tfbs_query = f"""
        query {{
          getTFBSByDatasetId(datasetId: "{ds_id}") {{
            sequence
            regulatedGenes {{
              name
            }}
          }}
        }}
        """
        bs_resp = requests.post(GRAPHQL_URL, json={"query": tfbs_query})
        bs_data = bs_resp.json()
        
        # Skip if there's an error
        if "errors" in bs_data:
            print(f"Error for dataset {ds_id}:", bs_data["errors"])
            continue
        
        tfbs_entries = bs_data["data"]["getTFBSByDatasetId"]
        if tfbs_entries:
            for entry in tfbs_entries:
                seq = entry["sequence"]
                # regulatedGenes is typically a list
                # If multiple genes per binding site, write them individually, or join with commas
                genes = entry["regulatedGenes"]
                if genes:
                    for gene_info in genes:
                        gene_name = gene_info["name"]
                        writer.writerow([ds_id, tf_name, gene_name, seq])
                else:
                    # no genes? you could still write a row
                    writer.writerow([ds_id, tf_name, "", seq])
        else:
            # No TFBS for this dataset
            writer.writerow([ds_id, tf_name, "", ""])
