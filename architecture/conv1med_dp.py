from torch import nn

in_channels = 6
out_channels = 1

class Conv1dMed_CHLA(nn.Module):
    def __init__(self, dp_rate):
        super(Conv1dMed_CHLA, self).__init__()
        self.dp_rate = dp_rate

        self.conv1 = nn.Conv1d(in_channels, 53, kernel_size=8, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(53)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(p=self.dp_rate)
        

        self.conv2 = nn.Conv1d(53, 77, kernel_size=8, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(77)
        self.af2 = nn.ReLU()
        self.do2 = nn.Dropout(p=self.dp_rate)


        self.deconv3 = nn.ConvTranspose1d(77, 46, kernel_size=7, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(46)
        self.af3 = nn.ReLU()  
        self.do3 = nn.Dropout(p=self.dp_rate)


        self.conv4 = nn.Conv1d(46, 128, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.af4 = nn.ReLU() 
        self.do4 = nn.Dropout(p=self.dp_rate)

        self.conv5 = nn.Conv1d(128, 98, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm1d(98)
        self.af5 = nn.ReLU() 
        self.do5 = nn.Dropout(p=self.dp_rate)

        self.deconv6 = nn.ConvTranspose1d(98, 12, kernel_size=2, stride=2, padding=1)
        self.do6 = nn.Dropout(p=self.dp_rate)

        self.flatten = nn.Flatten()

        self.fc7 = nn.Linear(600, 200)
        self.af7= nn.Softplus() # values must be positive


    def forward(self, x):
        x = self.af1(self.bn1(self.conv1(x)))
        x = self.do1(x)
        x = self.af2(self.bn2(self.conv2(x)))
        x = self.do2(x)
        x = self.af3(self.bn3(self.deconv3(x)))
        x = self.do3(x)
        x = self.af4(self.bn4(self.conv4(x)))
        x = self.do4(x)
        x = self.af5(self.bn5(self.conv5(x)))
        x = self.do5(x)
        x = self.deconv6(x)
        x = self.do6(x)
        x = self.flatten(x)
        x = self.af7(self.fc7(x))
        return x


class Conv1dMed_NITRATE(nn.Module):
    def __init__(self, dp_rate):
        super(Conv1dMed_NITRATE, self).__init__()
        self.dp_rate = dp_rate

        self.deconv1 = nn.ConvTranspose1d(in_channels, 61, kernel_size=10, stride=2, padding=1)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(p=self.dp_rate)

        self.deconv2 = nn.ConvTranspose1d(61, 128, kernel_size=2, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.af2 = nn.ReLU()
        self.do2 = nn.Dropout(p=self.dp_rate)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=2, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.af3 = nn.ReLU()  
        self.do3 = nn.Dropout(p=self.dp_rate)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.af4 = nn.ReLU()
        self.do4 = nn.Dropout(p=self.dp_rate)

        self.conv5 = nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=2)
        self.bn5 = nn.BatchNorm1d(128)
        self.af5 = nn.ReLU() 
        self.do5 = nn.Dropout(p=self.dp_rate)

        self.deconv6 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2, padding=2)
        self.bn6 = nn.BatchNorm1d(128)
        self.af6 = nn.ReLU()  
        self.do6 = nn.Dropout(p=self.dp_rate)

        self.conv7 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm1d(128)
        self.af7 = nn.ReLU() 
        self.do7 = nn.Dropout(p=self.dp_rate)

        self.conv8 = nn.Conv1d(128, 84, kernel_size=2, stride=1, padding=0)
        self.af8 = nn.ReLU() 
        self.do8 = nn.Dropout(p=self.dp_rate)

        self.conv9 = nn.Conv1d(84, 111, kernel_size=4, stride=1, padding=1)
        self.bn9 = nn.BatchNorm1d(111)
        self.af9 = nn.ReLU() 
        self.do9 = nn.Dropout(p=self.dp_rate)

        self.conv10 = nn.Conv1d(111, 1, kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()

        self.fc11 = nn.Linear(808, 200)
        self.af11= nn.Softplus() # values must be positive


    def forward(self, x):
        x = self.af1(self.deconv1(x))
        x = self.do1(x)
        
        x = self.af2(self.bn2(self.deconv2(x)))
        x = self.do2(x)
        
        x = self.af3(self.bn3(self.conv3(x)))
        x = self.do3(x)
        
        x = self.af4(self.bn4(self.conv4(x)))
        x = self.do4(x)
        
        x = self.af5(self.bn5(self.conv5(x)))
        x = self.do5(x)
        
        x = self.af6(self.bn6(self.deconv6(x)))
        x = self.do6(x)

        x = self.af7(self.bn7(self.conv7(x)))
        x = self.do7(x)

        x = self.af8(self.conv8(x))
        x = self.do8(x)

        x = self.af9(self.bn9(self.conv9(x)))
        x = self.do9(x)

        x = self.conv10(x) 
        x = self.flatten(x)
        
        x = self.af11(self.fc11(x))
        
        return x


def get_conv_model(variable, dp_rate):
    models = {
        "CHLA": Conv1dMed_CHLA,
        "NITRATE": Conv1dMed_NITRATE
    }

    try:
        model_class = models[variable.upper()]
    except KeyError:
        raise ValueError(
            f"Unknown target {variable!r}. Expected one of {list(models)}."
        )

    return model_class(dp_rate)

