## class name : Account 

class Account:
   ##  __init__ : constructor to initialize
    def __init__(self,account_number,account_holder,balance):
        self.acount_number=account_number
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


class SavingAccount(Account):
    def __init__(self, account_number, account_holder, balance,interest_rate,Bank_name):
        super().__init__(account_number, account_holder, balance)
        self.interst_rate=interest_rate
        self.Bank_name=Bank_name
    
    ## Polymorphism : 
    
