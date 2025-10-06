import synapseclient 
import synapseutils 
import os

from settings import SYS_TOKEN

syn = synapseclient.Synapse() 
syn.login(authToken=SYS_TOKEN) 

parent_id = "syn3193805"  # projeto raiz
pastas_desejadas = {
    "averaged-testing-images",
    "averaged-training-images",
    "averaged-training-labels",
}

BASE = os.path.join("data", parent_id)
os.makedirs(BASE, exist_ok=True)


for ch in syn.getChildren(parent_id):
    if ch["type"] == "folder" and ch["name"] in pastas_desejadas:
        destino = os.path.join(BASE, ch["name"])
        os.makedirs(destino, exist_ok=True)
        print(f"Baixando: {ch['name']} ({ch['id']}) → {destino}")
        synapseutils.syncFromSynapse(
            syn,
            ch["id"],
            path=destino,              
            ifcollision="overwrite.local",
            followLink=True
        )

