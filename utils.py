import torch

def getANM(V,E):
    V0,V1=V.index_select(1,E[:,0]),V.index_select(1,E[:,1])
    N=V1-V0
    M=(V1+V0)/2
    A=torch.sqrt((N**2).sum(dim=0).clamp_(min=1e-6))
    return A,N/A,M

def K(V1,N1,A1,V2,N2,A2):
    return (torch.exp(-64*((V1[:,:,None]-V2[:,None,:])**2).sum(0))*(1*((N1[:,:,None]*N2[:,None,:]).sum(dim=0))**2)*(A1[:,None]*A2[None,:])).sum()

def varifold(V1,E1):
    A1,N1,M1=getANM(V1,E1)
    cst=K(M1,N1,A1,M1,N1,A1)
    def loss(V2,E2):
        A2,N2,M2=getANM(V2,E2)
        return cst+K(M2,N2,A2,M2,N2,A2)-2*(K(M1,N1,A1,M2,N2,A2))
    return loss  

def SRNF(V1,V2,E):
    A1,N1,M1=getANM(V1,E)
    A2,N2,M1=getANM(V2,E)
    
    return ((N1*torch.sqrt(A1)-N2*torch.sqrt(A2))**2).sum()

def save3Dgraph(filename,V,E,colors=None): 
        V=V.cpu().numpy()
        E=E.cpu().numpy()
        file = open("{}.obj".format(filename), "w")
        lines=[]
        for i in range(0,V.shape[1]):
            lines.append("v ")
            for j in range(0,3):
                lines.append(str(V[j][i]))
                lines.append(" ")
            if colors is not None:
                for k in range(0,3):
                    lines.append(str(colors[k][i]))
                    lines.append(" ")                
            lines.append("\n")
        for i in range(0,E.shape[0]):
            lines.append("l ")
            for j in range(0, 2):
                lines.append(str(1+E[i][j]))
                lines.append(" ")
            lines.append("\n")
        file.writelines(lines)
        file.close()
                