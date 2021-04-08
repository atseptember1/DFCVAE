import os 
    
    
def progressBar(epoch, current, total, loss, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = "-" * int(percent/100 * barLength - 1) + ">"
    spaces  = "." * (barLength - len(arrow))
    print(f"Epoch {epoch}: [{arrow}{spaces}] {round(percent,2)}% - Loss: {loss}", end='\r')
    
def validate_path(path):
    if path is None:
        raise ValueError("Remember to assign the desirable parameter a path in config.py")
    
    if not(os.path.exists(path)):
        os.makedirs(path)