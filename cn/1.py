import ipaddress as ipa

def calculate_subnet(network,subnet_bits):
    try:
        net = ipa.IPv4Network(network)
        d = net.hosts()

        new_prefix = net.prefixlen + subnet_bits
        if new_prefix > 32:
            raise ValueError("Invalid subnet bits. Subnet prefix length cannot exceeed 32")
        
        subnets = list(net.subnets(new_prefix=new_prefix))

        print(f"Original Network : {network}")
        print(f"Original subnet mask : {net.netmask}")
        print(f"new subnet mask : {subnets[0].netmask}")
        print(f"number of subnets : {len(subnets)}")
        print(f"Hosts per subnet : {subnets[0].num_addresses-2}")
        print("\nSubnets:")
        for subnet in subnets:
            print(subnet)
        
    except ValueError as e:
        print(f"Error : {e}")

if __name__ =="__main__":
    network = "172.16.0.0/20"
    subnet_bits = 4
    calculate_subnet(network,subnet_bits)