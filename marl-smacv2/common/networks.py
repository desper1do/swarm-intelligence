"""
Нейросетевые архитектуры для MARL алгоритмов.
Поддержка MLP, RNN и трансформер-блоков.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MLP(nn.Module):
    """Простая многослойная перцептронная сеть."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        use_layer_norm: bool = False,
    ):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Выбор функции активации
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation) if output_activation else None

    def _get_activation(self, name: str):
        activations = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "leaky_relu": F.leaky_relu,
            "elu": F.elu,
            "sigmoid": torch.sigmoid,
        }
        return activations.get(name, F.relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            x = self.activation(x)

        x = self.output_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class RNNActor(nn.Module):
    """Рекуррентная сеть актора для агента."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        n_layers: int = 1,
    ):
        super(RNNActor, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_layers = n_layers

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, n_layers, batch_first=True)
        self.fc2 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Инициализирует скрытое состояние RNN."""
        return torch.zeros(self.n_layers, batch_size, self.rnn_hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, seq_len, obs_dim) или (batch, obs_dim)
            hidden_state: (n_layers, batch, rnn_hidden_dim)

        Returns:
            action_logits: (batch, seq_len, action_dim) или (batch, action_dim)
            new_hidden_state: (n_layers, batch, rnn_hidden_dim)
        """
        # Добавляем размерность последовательности если нужно
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
            squeeze_output = True
        else:
            squeeze_output = False

        x = F.relu(self.fc1(obs))
        x, h = self.rnn(x, hidden_state)
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)

        if squeeze_output:
            action_logits = action_logits.squeeze(1)

        return action_logits, h


class RNNCritic(nn.Module):
    """Рекуррентная сеть критика (централизованная)."""

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        n_layers: int = 1,
    ):
        super(RNNCritic, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_layers = n_layers

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, n_layers, batch_first=True)
        self.fc2 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.rnn_hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 2:
            state = state.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        x = F.relu(self.fc1(state))
        x, h = self.rnn(x, hidden_state)
        x = F.relu(self.fc2(x))
        value = self.value_head(x)

        if squeeze_output:
            value = value.squeeze(1)

        return value, h


class DQNNetwork(nn.Module):
    """Q-сеть для Value-based методов (QMIX агент)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.q_head(x)
        return q_values


class DRQNNetwork(nn.Module):
    """Deep Recurrent Q-Network для QMIX."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
    ):
        super(DRQNNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.rnn_hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        x = F.relu(self.fc1(obs))
        x, h = self.rnn(x, hidden_state)
        x = F.relu(self.fc2(x))
        q_values = self.q_head(x)

        if squeeze_output:
            q_values = q_values.squeeze(1)

        return q_values, h


class MLPActor(nn.Module):
    """MLP актор для MADDPG с LayerNorm для стабильности."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128],
        use_layer_norm: bool = True,
    ):
        super(MLPActor, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        self.output = nn.Linear(prev_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            x = F.relu(x)
        return torch.tanh(self.output(x))


class MLPCritic(nn.Module):
    """MLP критик для MADDPG (централизованный) с LayerNorm."""

    def __init__(
        self,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_dims: List[int] = [256, 128],
        use_layer_norm: bool = True,
    ):
        super(MLPCritic, self).__init__()
        input_dim = total_obs_dim + total_action_dim

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0]) if use_layer_norm else None

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        x = self.fc1(x)
        if self.ln1 is not None:
            x = self.ln1(x)
        x = F.relu(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            x = F.relu(x)
        return self.output(x)


class MixerNetwork(nn.Module):
    """
    QMIX mixing network.
    Комбинирует Q-значения отдельных агентов в общее Q-значение команды.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ):
        super(MixerNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Hypernetworks для генерации весов
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * hidden_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, hidden_dim),
        )

        # Hypernetworks для bias
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        agent_q_values: torch.Tensor,  # (batch, n_agents)
        state: torch.Tensor,  # (batch, state_dim)
    ) -> torch.Tensor:
        """
        Args:
            agent_q_values: Q-значения каждого агента
            state: глобальное состояние

        Returns:
            total_q_value: общее Q-значение команды (batch, 1)
        """
        batch_size = agent_q_values.size(0)
        agent_q_values = agent_q_values.view(batch_size, 1, self.n_agents)

        # Генерация первого слоя весов
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)

        b1 = self.hyper_b1(state).view(batch_size, 1, self.hidden_dim)

        # Первое преобразование
        hidden = F.elu(torch.bmm(agent_q_values, w1) + b1)

        # Генерация второго слоя весов
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.hidden_dim, 1)

        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # Второе преобразование
        total_q = torch.bmm(hidden, w2) + b2

        return total_q.view(batch_size, 1)


class AttentionCritic(nn.Module):
    """Критик с механизмом внимания для сложных сценариев."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
    ):
        super(AttentionCritic, self).__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(obs_dim, hidden_dim)
        self.key = nn.Linear(obs_dim, hidden_dim)
        self.value = nn.Linear(obs_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,  # (batch, n_agents, obs_dim)
        actions: torch.Tensor,  # (batch, n_agents, action_dim)
    ) -> torch.Tensor:
        # Self-attention по наблюдениям
        attn_out, _ = self.attention(obs, obs, obs)

        # Кодирование действий
        action_encoded = self.action_encoder(actions)

        # Объединение
        combined = torch.cat([attn_out, action_encoded], dim=-1)
        q_values = self.output(combined)

        return q_values  # (batch, n_agents, 1)


def init_weights(m: nn.Module):
    """Инициализация весов для стабильного обучения."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
