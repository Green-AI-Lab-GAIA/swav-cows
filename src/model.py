import numpy as np
import torch
from torch import nn
import torchvision

from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules.memory_bank import MemoryBankModule
    
class SwaV(nn.Module):
    def __init__(self, 
                 backbone_model='resnet50',
                input_size=11,
                n_hr_views=2, n_prototypes=60,
                n_features_swav=128, batch_size=64):
        super().__init__()

        backbone = self.resnet_backbone(backbone_model)

        with torch.no_grad():
            n_features_backbone = backbone(torch.randn(batch_size, 1, input_size[0], input_size[1])).shape[1]

        self.backbone = backbone

        self.projection_head = SwaVProjectionHead(
            input_dim=n_features_backbone, 
            hidden_dim=n_features_backbone, # arbitrary hidden dim, setting as the same as backbone
            output_dim=n_features_swav
        )
        
        self.prototypes = SwaVPrototypes(
            input_dim=n_features_swav, 
            n_prototypes=n_prototypes, 
            n_steps_frozen_prototypes=1
        )

        self.start_queue_at_epoch = 2
        
        self.queues = nn.ModuleList(
            [MemoryBankModule(size=(3840, n_features_swav)) for _ in range(n_hr_views)]
        )


    def resnet_backbone(self, backbone_model,pretrained_path=None):

        if backbone_model == 'resnet18':
            backbone = torchvision.models.resnet18()

        elif backbone_model == 'resnet50':
            backbone = torchvision.models.resnet50()

        #change to 1 channel
        original_conv1 = backbone.conv1

        backbone.conv1 = nn.Conv2d(
            in_channels=1,  # Change input channels to 1
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

        # Initialize new weights for the modified conv1
        nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Remove the fully connected layer (last layer) of ResNet
        backbone = nn.Sequential(*list(backbone.children())[:-1])

        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            backbone.load_state_dict(state_dict, strict=False) 
            print(f"Loaded pretrained weights from {pretrained_path}")

        return backbone


    def forward(self, high_resolution, low_resolution, epoch):
        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        high_resolution_prototypes = [
            self.prototypes(x, epoch) for x in high_resolution_features
        ]
        low_resolution_prototypes = [
            self.prototypes(x, epoch) for x in low_resolution_features
        ]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features, epoch)

        return high_resolution_prototypes, low_resolution_prototypes, queue_prototypes

    def _subforward(self, input):
        # with torch.no_grad():
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features, epoch):
        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f'The number of queues ({len(self.queues)}) should be equal to the number of high '
                f'resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly.'
            )

        # Get the queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)
            # Queue features are in (num_ftrs X queue_length) shape, while the high res
            # features are in (batch_size X num_ftrs). Swap the axes for interoperability.
            features = torch.permute(features, (1, 0))
            queue_features.append(features)

        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if self.start_queue_at_epoch > 0 and epoch < self.start_queue_at_epoch:
            return None

        # Assign prototypes
        queue_prototypes = [self.prototypes(x, epoch) for x in queue_features]
        return queue_prototypes


def swav_train(model,  dataloader, transform, criterion, optimizer, params, checkpoint_folder, log_file):
    patience, epoch = 0, 0
    min_loss = np.inf
    training_params = params['training']
    data_params = params['data']

    print(f'Starting Training')
    while True:
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(training_params['device'])
            views = transform(batch)
            high_resolution, low_resolution = views[:data_params['n_hr_views']], views[data_params['n_hr_views']:]

            high_resolution, low_resolution, queue = model(
                high_resolution, low_resolution, epoch
            )

            loss = criterion(high_resolution, low_resolution, queue)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        patience += 1
        epoch += 1
        loss_print = f'epoch: {epoch:>04}, loss: {avg_loss:.5f}'

        if avg_loss < min_loss:
            min_loss = avg_loss
            loss_print += ' (*)'
            patience = 0
            torch.save(model.state_dict(), f'{checkpoint_folder}/minloss.pth')
        
        if (epoch % 100) == 0:
            torch.save(model.state_dict(), f'{checkpoint_folder}/epoch_{epoch}.pth')
        
        print(loss_print)
        log_file.write(loss_print + "\n")

        if patience >= training_params['max_patience'] and epoch >= training_params['min_epochs']:
            torch.save(model.state_dict(), f'{checkpoint_folder}/epoch_{epoch}.pth')
            break