class calculatorClass():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def add(self):
        return self.x + self.y
    
    def divide(self):
        if(self.y != 0):
            return self.x / self.y
        else:
            __invalidInputPrint(x, "division")
            
    def __invalidInputPrint(x, operantType):
        print(f"Invalid input parameter {x} for ", operantType)