from data.profiler import profile

for name in ["HuggingFaceFW/fineweb","HuggingFaceFW/fineweb-edu"]:
    try: print(profile(name,n=1000))
    except Exception as e: print(name,"ERROR:",e)
