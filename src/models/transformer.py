import torch
import torch.nn as nn

from config import(
    IN_CHANNELS,
    CONV_OUT,
    D_MODEL,
    NHEAD,
    NUM_LAYERS,
    NUM_CLASSES,
    DIM_FEEDFORWARD,
    DROPOUT
)

class MultiScaleConv(nn.Module):
    """
    Extract temporal EEG patterns at multiple scales.
    
    Small kernel: 
        capture short signal changes
        
    Medium kernel:
        captures medium temporal patterns
        
    Large kernel:
        captures slower EEG dynamics
        
    """
    
    def __init__(
        self,
        in_channels = IN_CHANNELS,
        out_channels = CONV_OUT
    ):
        super().__init__()
        
        self.conv_small = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = 5,
            padding = 2
        )
        
        self.conv_medium = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = 25,
            padding = 12
        )
        
        self.conv_large = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = 75,
            padding = 37
        )
        
        
    def forward(self, x):
        # input
        # (batch, time, channels)
        
        # Conv1d expects:
        # (batch, channels, time)
        
        x = x.transpose(1, 2)
        
        small = self.conv_small(x)
        medium = self.conv_medium(x)
        large = self.conv_large(x)
        
        # combine feature detectors
        x = torch.cat(
            [small, medium, large],
            dim = 1 # join the feature maps together ( 32 + 32 + 32)
        )     
        
        # back to Transformer format
        # (batch, time, features)
        
        x = x.transpose(1, 2)
        return x   
    
    
class ClassificationHead(nn.Module):
    """
    Converts Transformer features into final class predictions.
    
    input:
        (batch, time, features)
        
    output:
        (batch, number_of_classes)
        
    """
    
    def __init__(
        self,
        d_model = D_MODEL,
        num_classes = NUM_CLASSES
    ):
        
        super().__init__()
        
        self.linear = nn.Linear(
            d_model,
            num_classes
        )
        
    def forward(self, x):
         # x shape:
         # (batch, time, features)
         # eg: (16, 641, 96)
         x = x.mean(dim = 1)
         
         # now
         # (16, 96)
         
         x = self.linear(x)
         
         # now:
         # (16, 2)
         return x
    
    
class EEGTransformer(nn.Module):
    """
    Complete end-to-end EEG classification pipeline.

    Input:  (batch, 641, 64) — EEG signal, time-first
    Output: (batch, 2)       — left/right class scores

    Pipeline:
        MultiScaleConv     → (batch, 641, 96)  local patterns
        TransformerEncoder → (batch, 641, 96)  global relationships
        ClassificationHead → (batch, 2)        final decision

    
    """
    def __init__(
        self,
        in_channels = IN_CHANNELS,
        conv_out = CONV_OUT,
        d_model = D_MODEL,
        nhead = NHEAD,
        num_layers = NUM_LAYERS,
        num_classes = NUM_CLASSES
    ):
        super().__init__()
        
        # 1.0 multi-scale CNN
        self.multi_sclae_conv = MultiScaleConv(
            in_channels = in_channels,
            out_channels = conv_out
        )
        
        # 2.0 transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = DIM_FEEDFORWARD,
            batch_first = True,
            dropout = DROPOUT
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers
        )
        
        # 3.0 classification head
        self.head = ClassificationHead(
            d_model = d_model,
            num_classes = num_classes
        )
        
    def forward(self, X):
        x = self.multi_sclae_conv(x)    # local pattern extraction
        x = self.transformer(x)         # global relationship detection
        x = self.head(x)                # classify left or right
        
        
        
def load_model(model_path, config_path = None, device = None):
    """
    loads the trained EEGTransformer from saved weights.
    used by the serving layer (predict.py) to load the model once
    at server startup.
    then reusing for all incoming inference requests.
    
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # build the model architecture
    model = EEGTransformer()
    
    # load the trained weights into the architecture 
    # map_location ensures weights load correctly
    # regardless of whether they were saved on GPU or CPU
    model.load_state_dict(
        torch.load(model_path, map_location = device)
        
        
    )
    
    # set to evaluation mode
    # this disables dropout (which only runs during training)
    # so predictions are consistent and deterministic
    model.eval()
    model.to(device)
    
    print(f"Model loaded from {model_path}")
    print(f"Running on: {device}")
    
    return model, device        
    