import socket as s 

def dns_lookup(ip_address=None,domain_name=None):
    try:
        if ip_address:
            result  = s.gethostbyaddr(ip_address)
            return f"Ip Address : {ip_address} -> Domain Name : {result[0]}"
        
        elif domain_name:
            result = s.gethostbyname(domain_name)
            return f"\nDomain Name : {domain_name} -> IP address : {result}"
        
        else:
            return "Please provide either an IP Address or a domain name"

    except s.herror as e:
        return f"Error : {e}"
    
if __name__ == "__main__":
    print("DNS Lookup program")

choice = input("Do you want to lookup by (1)IP Address or (2)Domain Name ? Enter 1 or 2 :")
if choice=="1":
    ip_address = input("Enter the IP address : ")
    result = dns_lookup(ip_address=ip_address)
    print(result)

elif choice=="2":
    domain_name = input("Enter the domain name : ")
    result = dns_lookup(domain_name=domain_name)
    print(result)

else:
    print("invalid choice")