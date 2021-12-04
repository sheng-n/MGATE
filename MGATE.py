import torch
import torch.nn as nn
import torch.nn.functional as F




'''Graph convolution'''

class GraphConvolution(nn.Module):

    def __init__(self, in_size, out_size,):
        super(GraphConvolution, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))
        nn.init.xavier_uniform_(self.weight.data, gain = 1.414)

    def forward(self, adj, features):

       out = torch.mm(adj, features)   # A*X
       out = torch.mm(out,self.weight)    # A*X*W

       return out

'''Graph convolution layer'''

class GCNNet(nn.Module):
    """
    two layer graph convolution
    """

    def __init__(self, input_size, hidden_ize, n_class_size):
        super(GCNNet, self).__init__()

        self.gcn1 = GraphConvolution(input_size, hidden_ize)
        self.gcn2 = GraphConvolution(hidden_ize, n_class_size)

    def forward(self, adj, features):
        out = F.relu(self.gcn1(adj, features), inplace = False)  # 1140*700
        # print("out1: ",out.shape)
        out = self.gcn2(adj, out)  # 1140*500

        return out


'''Complex graph encoder'''

class GCN_complex(nn.Module):

    def __init__(self, input_c, hidden_c, n_class_c):
        super(GCN_complex, self).__init__()

        self.complex = GCNNet(input_c, hidden_c, n_class_c)

    def forward(self, adj_c, features_c):

        out = self.complex(adj_c, features_c)

        return out


'''Intra graph encoder: intra_L, intra_M, intra_D'''

class GCN_intra(nn.Module):

    def __init__(self, input_l, input_m, input_d, hidden_intra, n_class_intra):
        super(GCN_intra, self).__init__()


        # intra_lncRNA graph
        self.gcn_intra_l = GCNNet(input_l, hidden_intra, n_class_intra)

        # intra_miRNA graph
        self.gcn_intra_m = GCNNet(input_m, hidden_intra, n_class_intra)

        # intra_disease graph
        self.gcn_intra_d = GCNNet(input_d, hidden_intra, n_class_intra)

    def forward(self, adj_l, features_l, adj_m, features_m, adj_d, features_d):

        out_l = self.gcn_intra_l(adj_l, features_l)
        out_m = self.gcn_intra_m(adj_m, features_m)
        out_d = self.gcn_intra_d(adj_d, features_d)

        return out_l, out_m, out_d


'''Inter graph encoder: intra_LD, intra_LM, intra_MD'''

class GCN_inter(nn.Module):

    def __init__(self, input_ld, input_lm, input_md, hidden_inter, n_class_inter):
        super(GCN_inter, self).__init__()

        # inter lncRNA-disease graph
        self.gcn_inter_ld = GCNNet(input_ld, hidden_inter, n_class_inter)

        # inter lncRNA-miRNA graph
        self.gcn_inter_lm = GCNNet(input_lm, hidden_inter, n_class_inter)

        # inter miRNA-disease graph
        self.gcn_inter_md = GCNNet(input_md, hidden_inter, n_class_inter)

    def forward (self, adj_ld, features_ld, adj_lm, features_lm, adj_md, features_md) :

        out_ld = self.gcn_inter_ld(adj_ld, features_ld)
        out_lm = self.gcn_inter_lm(adj_lm, features_lm)
        out_md = self.gcn_inter_md(adj_md, features_md)

        return out_ld, out_lm, out_md


'''Multi_graph attention fusion'''

class Attention_fusion(nn.Module):

    def __init__(self, input_att, output_att):
        super(Attention_fusion,self).__init__()

        self.W_parameters = nn.Parameter(torch.rand(input_att, output_att))
        nn.init.xavier_normal_(self.W_parameters)

        self.h_n_parameters = nn.Parameter(torch.randn(1, input_att))
        nn.init.xavier_normal_(self.h_n_parameters)

    def forward(self, x1,x2,x3):

        # reslut = torch.cat((x1, x2, x3), dim=1)  # 1,2,1140
        # x:1140*768

        temp_node_1 = torch.tanh(torch.matmul(self.W_parameters,x1))
        temp_node_2 = torch.tanh(torch.matmul(self.W_parameters, x2))
        temp_node_3 = torch.tanh(torch.matmul(self.W_parameters, x3))
        # print("W_out_size :", temp_node_1.size(),temp_node_2.size(),temp_node_3.size())     # d*d  256*256

        node_score_1 = torch.mm(self.h_n_parameters,temp_node_1)
        node_score_2 = torch.mm(self.h_n_parameters, temp_node_2)
        node_score_3 = torch.mm(self.h_n_parameters, temp_node_3)
        # print("h_out_size  :",node_score_1.size(),node_score_2.size(),node_score_3.size())  #1,d  1,256

        nodes_score = torch.cat((node_score_1,node_score_2,node_score_3), dim = 0) #3,256
        # print("nodes_score: ",nodes_score.shape)

        # nodes_score = torch.cat((node_score_1,node_score_2,node_score_3),dim=1)
        # print("nodes_score: ", nodes_score.shape)
        beta = F.softmax(nodes_score, dim=0)  # 3,256
        # print(beta[3].size())
        # print("beta: ", beta[0].size(),x1[0].size())

        x_l = torch.zeros((1140,256))
        x_d = torch.zeros((1140, 256))
        x_m = torch.zeros((1140, 256))
        for i in range(1140):
            x_l[i] = torch.mul(beta[0], x1[i])
            x_d[i] = torch.mul(beta[1], x2[i])
            x_m[i] = torch.mul(beta[2], x3[i])

        # z_i = torch.matmul(beta, x)  # 1 * 1 * 1140
        # print("zi: ", z_i.shape)

        out = x_l + x_d + x_m
        # print(x1.size(),x2.size(),x3.size())

        return out


'''Decoder'''
class Decoder(nn.Module):

    def __init__(self, out_desize):
        super(Decoder, self).__init__()

        self.de_weight = nn.Parameter(torch.FloatTensor(out_desize, out_desize))
        nn.init.xavier_uniform_(self.de_weight.data, gain = 1.414)

    def forward(self, x):
        z = torch.mm(x, self.de_weight)  # A*W
        y = x.permute(1, 0)  # dim convert
        z = torch.mm(z, y)  # A*W*AT
        z = torch.sigmoid(z)
        # z = torch.Tensor.relu(z)
        return z


'''Intra graph decoder: intra_L, intra_M, intra_D'''

class MGATE_decoder(nn.Module):

    def __init__(self, decoder_size):
        super(MGATE_decoder, self).__init__()

        # complex_graph
        self.decoder_complex = Decoder(decoder_size)

        # intra_lncRNA graph
        self.decoder_intra_l = Decoder(decoder_size)

        # intra_miRNA graph
        self.decoder_intra_m = Decoder(decoder_size)

        # intra_disease graph
        self.decoder_intra_d = Decoder(decoder_size)

        # inter lncRNA-disease graph
        self.decoder_inter_ld = Decoder(decoder_size)

        # inter lncRNA-miRNA graph
        self.decoder_inter_lm = Decoder(decoder_size)

        # inter miRNA-disease graph
        self.decoder_inter_md = Decoder(decoder_size)

        # cross fusion embedding
        self.decoder_cross_em = Decoder(decoder_size)


    def forward(self, out_c, out_l, out_m, out_d, out_ld, out_lm, out_md, out_em):

        de_out_c = self.decoder_complex(out_c)

        de_out_l = self.decoder_intra_d(out_l)
        de_out_m = self.decoder_intra_m(out_m)
        de_out_d = self.decoder_intra_d(out_d)

        de_out_ld = self.decoder_inter_ld(out_ld)
        de_out_lm = self.decoder_inter_lm(out_lm)
        de_out_md = self.decoder_inter_md(out_md)

        de_out_em = self.decoder_cross_em(out_em)

        return de_out_c, de_out_l, de_out_m, de_out_d, de_out_ld, de_out_lm, de_out_md, de_out_em




'''MGATE'''

class MGATE(nn.Module):

    def __init__(self, input_c, hidden_c, n_class_c, input_l, input_m, input_d, hidden_intra, n_class_intra,
                 input_ld, input_lm, input_md, hidden_inter, n_class_inter, input_att, output_att, decoder_size):
        super(MGATE, self).__init__()

        """
        hidden_c = hidden_intra = hidden_inter = 700
        n_class_c = n_class_intra = n_class_inter = 256
        
        """


        # Encoder
        self.gcn_complex = GCN_complex(input_c, hidden_c, n_class_c)
        self.gcn_intra = GCN_intra(input_l, input_m, input_d, hidden_intra, n_class_intra)
        self.gcn_inter = GCN_inter(input_ld, input_lm, input_md, hidden_inter, n_class_inter)

        # attention
        self.attention = Attention_fusion(input_att, output_att)

        # decoder: 8
        self.decode = MGATE_decoder(decoder_size)


    def forward(self, adj_c, features_c, adj_l, features_l, adj_m, features_m, adj_d, features_d,
                adj_ld, features_ld, adj_lm, features_lm, adj_md, features_md):

        out_c = self.gcn_complex(adj_c, features_c)       # 1140 * out_dim
        # print("complex_graph: ", out_c.shape)

        out_l, out_m, out_d = self.gcn_intra(adj_l, features_l, adj_m, features_m, adj_d, features_d)

        out_ld, out_lm, out_md = self.gcn_inter(adj_ld, features_ld, adj_lm, features_lm, adj_md, features_md)


        out_intra = torch.cat((out_l,out_d,out_m), dim = 0)

        out_inter_1 = torch.cat((out_ld[:240],out_ld[240:],out_lm[240:]),dim = 0)
        out_inter_2 = torch.cat((out_lm[:240],out_md[495:],out_md[:495]),dim=0)
        out_inter = (out_inter_1 + out_inter_2)/2
        # print("intra_graph: ", out_inter_1.shape,out_inter_2.shape)

        out_cross_fusion = self.attention(out_c, out_intra, out_inter).cuda()
        # print("out_cross :",out_Cross.size())

        # out_Cross = (out_C + out_intra + out_inter_1 + out_inter_2)/4
        de_out_c, de_out_l, de_out_m, de_out_d, de_out_ld, de_out_lm, de_out_md, de_out_em = self.decode(out_c, out_l,
                           out_m, out_d, out_ld, out_lm, out_md, out_cross_fusion)


        return de_out_c, de_out_l, de_out_m, de_out_d, de_out_ld, de_out_lm, de_out_md, de_out_em, out_cross_fusion


