'''

import os

path="notebooks\research.ipynb"

dir,file=os.path.split(path)

os.makedirs(dir_exit_ok=True)

with open(path,"w") as f:
    pass
  
    
    '''

from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData

custdataobj=CustomData(1.52,62.2,58.0,7.27,7.33,4.55,"Premimum","F","VS2")

data=custdataobj.get_data_as_dataframe()

print(data)