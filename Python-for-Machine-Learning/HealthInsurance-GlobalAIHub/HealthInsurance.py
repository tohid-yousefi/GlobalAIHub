#Define the "HealthInsurance" class
class HealthInsurance:
    #Initialize the object's attributes
    def __init__(self,company_name,foundation_year,founder_name,company_slogan,num_of_employees,num_of_clients):
      self.company_name = company_name
      self.foundation_year = foundation_year
      self.founder_name = founder_name
      self.company_slogan = company_slogan
      self.num_of_employees = num_of_employees
      self.num_of_clients = num_of_clients

    #Define the print_report method
    def print_report(self):
      print(f"""The company {self.company_name} was founded in {self.foundation_year}. 
      The founder of company is {self.founder_name}.
      Company slogan: {self.company_slogan}
      Number of Employees: {self.num_of_employees}
      Number of Clients: {self.num_of_clients}""")

    #Define the sup_health_insurance method
    def sup_health_insurance(self,age,chronic_disease,income):
      #if-else statements to check whether person can get supplemental insurance or not
      if age > 60 and chronic_disease == True and income < 6000:
        print("We ar sorry! You are not eligable for supplemental health insurance.")
      elif age < 60 and income >=6000 or chronic_disease == False:
        print("Congratulations! You can get supplemental health insurance.")

    #Define the update_num_clients method
    def update_num_clients(self,new_numnber):
      self.num_of_clients = new_numnber
      print(f"Number of clients has been changed to {self.num_of_clients} !")
   
#Create the object "HI_company1" with it's attributes
HI_company1 = HealthInsurance("Healthy",2012,"Bob Mayer","We care for you",3500,13230)


#Use the sup_health_insurance for the new customer
HI_company1.sup_health_insurance(45,False,5000)

#Use the sup_health_insurance for the new customer
HI_company1.sup_health_insurance(45,False,5000)

#Update the number of clients
HI_company1.update_num_clients(13231)

#Call the print_report method for HI_company1
HI_company1.print_report()

