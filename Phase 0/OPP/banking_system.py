## class name : Account 

class Account:
   ##  __init__ : constructor to initialize
    def __init__(self,account_number,account_holder,balance):
        self.account_number=account_number
        self.account_holder=account_holder
        self.balance=balance
    ## self : used to represent the curretn instant object.

    def deposit(self,amount):
        """ Deposite money into account"""
        if(amount>0):
            self.balance+=amount
            print(f"Deposite {amount} amd current amount be {self.balance}")
        else:
            print("Deposite amount be positive")

     ## withdraw the amount   
    def withdrow(self,amount):
        """ Withdraw the amount"""
        if(self.balance>=amount and amount>0):
            self.balance-=amount
            print(f"withdraw the amount {amount} and updated current amount be  {self.balance}")
        else:
            print("Please withdraw positive amount")
        
    ## display details 
    def DisplayDetail(self):
        print(f"Account number : {self.acount_number}, account holder : {self.account_holder} and account balance : {self.balance}")

## Inheritanc : Saving account inherits form account

    ## Inheritance ->
class SavingAccount(Account):
    def __init__(self, account_number, account_holder, balance,bank_name,interest_rate):
        super().__init__(account_number, account_holder, balance)
        self.interest_rate=interest_rate
        self.bank_name=bank_name
    
    ## Polymorphism : 
    def display(self):
        print(f"Saving account number :{self.account_number}, balance: {self.balance} ")
    
    ## Added the interest in the saving account
    def add_interest(self):
        interest=self.balance*(self.interest_rate/100)
        self.balance+=interest
        print(f"the interest rate {interest}")


class CurrentAccount(Account):
    def __init__(self,account_number,account_holder,balance,overdraft_limit):
        super().__init__(account_number,account_holder,balance)
        self.overdraft_limit=overdraft_limit

    ## polymorphism overriding withdraw method

    def withdrow(self, amount):
        if(amount<=self.balance+self.overdraft_limit):
            self.balance-=amount
            print(f"withdraw amount {amount} and current balance {self.balance}")
        
        else:
            print("Overdraf limit over")

    def display(self):
        print(f"Current account number :{self.account_number}, balance: {self.balance} and user name {self.account_holder} ")
    

if __name__=='__main__':
    acc1=SavingAccount('RK252627','Rishabh kushwaha',99999,'PNB',3)
    acc1.display()
    acc1.withdrow(455)
    acc1.add_interest()
    acc1.deposit(78545)

    acc2=CurrentAccount('HM456','Happy Maurya',45612,5000)
    acc2.display()
    acc2.withdrow(45615)
    





