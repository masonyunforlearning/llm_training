$dirs=@("configs","data","tokenizer","models","training","scripts","experiments","experiments/proxy","experiments/final_1.2B","docs")
foreach($d in $dirs){New-Item -ItemType Directory -Force -Path $d | Out-Null}
Write-Host "Directory structure created."
