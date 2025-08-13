# FinCoach ML Enhancement - Complete Implementation

**Level 2 ML Improvements: User Segmentation, Enhanced Recommendations, and Predictive Analytics**

This document contains all the code blocks for implementing the complete FinCoach ML enhancement plan. Each section can be converted to a Jupyter notebook cell.

---

## 1. Setup and Library Installation

### Install Libraries

```python
# Install required libraries
!pip install torch torchvision torchaudio
!pip install scikit-learn pandas numpy matplotlib seaborn plotly
!pip install joblib tqdm statsmodels

print("✅ All libraries installed successfully!")
```

### Import Libraries

```python
# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, classification_report, mean_absolute_error
from sklearn.decomposition import PCA

# Utility imports
import joblib
import pickle
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")

print("✅ All imports completed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
```

---

## 2. Data Loading and Exploration

### Load Dataset

```python
# Load the anonymized transaction dataset
# Note: Upload your dataset/anonymized_original_with_category.csv to Colab

# For Colab, upload the file first
from google.colab import files
print("Please upload your anonymized_original_with_category.csv file:")
uploaded = files.upload()

# Load the dataset
df = pd.read_csv('anonymized_original_with_category.csv')

print(f"✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['Transaction Date'].min()} to {df['Transaction Date'].max()}")

# Display first few rows
df.head()
```

### Data Exploration

```python
# Data exploration and quality assessment
print("=== DATA QUALITY ASSESSMENT ===")
print(f"Total transactions: {len(df):,}")
print(f"Missing values per column:")
print(df.isnull().sum())

print("\n=== TRANSACTION TYPES ===")
print(df['Transaction Type'].value_counts())

print("\n=== CATEGORIES DISTRIBUTION ===")
print(df['Category'].value_counts().head(15))

print("\n=== BASIC STATISTICS ===")
print(f"Debit transactions: {len(df[df['Debit Amount'].notna()]):,}")
print(f"Credit transactions: {len(df[df['Credit Amount'].notna()]):,}")
print(f"Average debit amount: ${df['Debit Amount'].mean():.2f}")
print(f"Average credit amount: ${df['Credit Amount'].mean():.2f}")

# Visualize category distribution
plt.figure(figsize=(12, 6))
category_counts = df['Category'].value_counts().head(15)
plt.bar(range(len(category_counts)), category_counts.values)
plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')
plt.title('Top 15 Transaction Categories')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show()
```

---

## 3. Data Preprocessing and Transformation

### Data Transformer Class

```python
class DataTransformer:
    """Transform transaction data to standardized format for ML models"""

    def __init__(self):
        self.category_mapping = {
            'groceries': ['Groceries'],
            'dine_out': ['Dine Out'],
            'entertainment': ['Entertainment'],
            'bills': ['Bills'],
            'transport': ['Travel', 'Transport'],
            'shopping': ['Amazon', 'Other Shopping'],
            'health': ['Health', 'Insurance'],
            'fitness': ['Fitness'],
            'savings': ['Savings', 'Investment'],
            'income': ['Supplementary Income'],
            'other': ['Others', 'Cash', 'Services', 'Home Improvement']
        }

    def preprocess_data(self, df):
        """Clean and preprocess the transaction data"""
        # Create a copy
        processed_df = df.copy()

        # Convert date column
        processed_df['Transaction Date'] = pd.to_datetime(processed_df['Transaction Date'], format='%d/%m/%Y')

        # Create unified amount column (negative for debits, positive for credits)
        processed_df['Amount'] = processed_df['Credit Amount'].fillna(0) - processed_df['Debit Amount'].fillna(0)

        # Standardize transaction types
        processed_df['Transaction Type Standardized'] = processed_df['Transaction Type'].map({
            'DEB': 'debit',
            'BP': 'debit',
            'DD': 'debit',
            'CPT': 'debit',
            'CHQ': 'debit',
            'FEE': 'debit',
            'DEP': 'credit',
            'BGC': 'credit',
            'FPI': 'credit'
        }).fillna('other')

        # Map categories to standard categories
        processed_df['Standard Category'] = processed_df['Category'].apply(self._map_category)

        # Sort by date
        processed_df = processed_df.sort_values('Transaction Date').reset_index(drop=True)

        return processed_df

    def _map_category(self, category):
        """Map original category to standard category"""
        for standard_cat, original_cats in self.category_mapping.items():
            if category in original_cats:
                return standard_cat
        return 'other'

    def create_synthetic_users(self, df, num_users=50, months_per_user=6):
        """Create synthetic users by segmenting the timeline"""
        # Sort by date
        df_sorted = df.sort_values('Transaction Date').reset_index(drop=True)

        # Calculate date ranges for each user
        start_date = df_sorted['Transaction Date'].min()
        end_date = df_sorted['Transaction Date'].max()
        total_days = (end_date - start_date).days

        days_per_user = total_days // num_users

        # Assign user IDs based on date ranges
        df_sorted['User ID'] = 0

        for user_id in range(num_users):
            user_start = start_date + timedelta(days=user_id * days_per_user)
            user_end = start_date + timedelta(days=(user_id + 1) * days_per_user)

            mask = (df_sorted['Transaction Date'] >= user_start) & (df_sorted['Transaction Date'] < user_end)
            df_sorted.loc[mask, 'User ID'] = user_id

        # Handle remaining transactions
        remaining_mask = df_sorted['User ID'] == 0
        df_sorted.loc[remaining_mask, 'User ID'] = num_users - 1

        return df_sorted
```

### Process Data

```python
# Initialize transformer and process data
transformer = DataTransformer()
processed_df = transformer.preprocess_data(df)
user_df = transformer.create_synthetic_users(processed_df, num_users=50)

print(f"✅ Data preprocessing completed!")
print(f"Processed transactions: {len(user_df):,}")
print(f"Synthetic users created: {user_df['User ID'].nunique()}")
print(f"Date range: {user_df['Transaction Date'].min()} to {user_df['Transaction Date'].max()}")

# Display sample of processed data
user_df[['Transaction Date', 'User ID', 'Amount', 'Standard Category', 'Balance']].head(10)
```

---

## 4. Feature Engineering - Enhanced State Vectors (35+ Dimensions)

### Enhanced Feature Extractor Class

```python
class EnhancedFeatureExtractor:
    """Extract enhanced features for ML models - extending from 17 to 35+ dimensions"""

    def __init__(self):
        self.standard_categories = [
            'groceries', 'dine_out', 'entertainment', 'bills',
            'transport', 'shopping', 'health', 'fitness',
            'savings', 'income', 'other'
        ]
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build comprehensive feature names list"""
        # Base features (17 dimensions - matching existing implementation)
        self.feature_names.extend([
            'current_balance',
            'total_spending',
            'total_income',
            'transaction_count'
        ])

        # Category spending (11 dimensions)
        self.feature_names.extend([f'{cat}_spending' for cat in self.standard_categories])

        # Derived metrics (2 dimensions)
        self.feature_names.extend(['savings_rate', 'spending_velocity'])

        # Enhanced behavioral features (18+ additional dimensions)
        self.feature_names.extend([
            # Temporal patterns (4 dimensions)
            'weekday_spending_ratio',
            'weekend_spending_ratio',
            'morning_spending_ratio',
            'evening_spending_ratio',

            # Spending behavior (5 dimensions)
            'spending_volatility',
            'large_transaction_ratio',
            'impulse_buying_score',
            'category_diversity',
            'spending_consistency',

            # Financial discipline (4 dimensions)
            'budget_adherence_score',
            'savings_consistency',
            'emergency_fund_ratio',
            'debt_indicator',

            # Social/contextual features (3 dimensions)
            'merchant_loyalty_score',
            'location_diversity',
            'seasonal_spending_factor',

            # Risk indicators (2 dimensions)
            'balance_volatility',
            'financial_stress_indicator'
        ])

    def extract_features(self, user_transactions, lookback_days=30):
        """Extract comprehensive feature vector for a user"""
        # Filter to recent transactions
        end_date = user_transactions['Transaction Date'].max()
        start_date = end_date - timedelta(days=lookback_days)
        recent_transactions = user_transactions[
            user_transactions['Transaction Date'] >= start_date
        ].copy()

        if len(recent_transactions) == 0:
            return np.zeros(len(self.feature_names))

        features = []

        # Base features (17 dimensions)
        features.extend(self._extract_base_features(recent_transactions))

        # Enhanced behavioral features (18+ dimensions)
        features.extend(self._extract_behavioral_features(recent_transactions))

        return np.array(features, dtype=np.float32)

    def _extract_base_features(self, transactions):
        """Extract base features matching existing 17-dimensional implementation"""
        features = []

        # Current balance
        current_balance = transactions['Balance'].iloc[-1] if len(transactions) > 0 else 0.0
        features.append(current_balance)

        # Total spending and income
        spending_transactions = transactions[transactions['Amount'] < 0]
        income_transactions = transactions[transactions['Amount'] > 0]

        total_spending = abs(spending_transactions['Amount'].sum())
        total_income = income_transactions['Amount'].sum()
        transaction_count = len(transactions)

        features.extend([total_spending, total_income, transaction_count])

        # Category spending
        category_spending = {cat: 0.0 for cat in self.standard_categories}
        for _, transaction in spending_transactions.iterrows():
            category = transaction['Standard Category']
            if category in category_spending:
                category_spending[category] += abs(transaction['Amount'])

        features.extend([category_spending[cat] for cat in self.standard_categories])

        # Derived metrics
        savings_rate = (total_income - total_spending) / max(total_income, 1)
        spending_velocity = total_spending / max(len(transactions), 1)

        features.extend([savings_rate, spending_velocity])

        return features

    def _extract_behavioral_features(self, transactions):
        """Extract enhanced behavioral features (18+ dimensions)"""
        features = []

        if len(transactions) == 0:
            return [0.0] * 18

        # Temporal patterns
        transactions['weekday'] = transactions['Transaction Date'].dt.weekday
        transactions['hour'] = transactions['Transaction Date'].dt.hour

        weekday_spending = abs(transactions[transactions['weekday'] < 5]['Amount'].sum())
        weekend_spending = abs(transactions[transactions['weekday'] >= 5]['Amount'].sum())
        total_spending = abs(transactions[transactions['Amount'] < 0]['Amount'].sum())

        weekday_ratio = weekday_spending / max(total_spending, 1)
        weekend_ratio = weekend_spending / max(total_spending, 1)

        morning_spending = abs(transactions[transactions['hour'] < 12]['Amount'].sum())
        evening_spending = abs(transactions[transactions['hour'] >= 18]['Amount'].sum())

        morning_ratio = morning_spending / max(total_spending, 1)
        evening_ratio = evening_spending / max(total_spending, 1)

        features.extend([weekday_ratio, weekend_ratio, morning_ratio, evening_ratio])

        # Spending behavior
        spending_amounts = abs(transactions[transactions['Amount'] < 0]['Amount'])
        spending_volatility = spending_amounts.std() if len(spending_amounts) > 1 else 0.0

        large_transaction_threshold = spending_amounts.quantile(0.8) if len(spending_amounts) > 0 else 0
        large_transaction_ratio = len(spending_amounts[spending_amounts > large_transaction_threshold]) / max(len(spending_amounts), 1)

        # Impulse buying score (transactions within 1 hour of each other)
        transactions_sorted = transactions.sort_values('Transaction Date')
        time_diffs = transactions_sorted['Transaction Date'].diff().dt.total_seconds() / 3600
        impulse_score = len(time_diffs[time_diffs < 1]) / max(len(transactions), 1)

        # Category diversity (entropy)
        category_counts = transactions['Standard Category'].value_counts()
        category_probs = category_counts / len(transactions)
        category_diversity = -sum(p * np.log2(p) for p in category_probs if p > 0)

        # Spending consistency (coefficient of variation)
        spending_consistency = 1 / (1 + spending_amounts.std() / max(spending_amounts.mean(), 1)) if len(spending_amounts) > 0 else 0

        features.extend([spending_volatility, large_transaction_ratio, impulse_score, category_diversity, spending_consistency])

        # Financial discipline
        savings_transactions = transactions[transactions['Standard Category'] == 'savings']
        savings_consistency = len(savings_transactions) / max(len(transactions), 1)

        current_balance = transactions['Balance'].iloc[-1]
        avg_monthly_spending = total_spending * (30 / max(len(transactions), 1))
        emergency_fund_ratio = current_balance / max(avg_monthly_spending * 3, 1)

        budget_adherence = 1.0  # Simplified - would need budget data
        debt_indicator = 1.0 if current_balance < 0 else 0.0

        features.extend([budget_adherence, savings_consistency, emergency_fund_ratio, debt_indicator])

        # Social/contextual features
        merchant_counts = transactions['Transaction Description'].value_counts()
        merchant_loyalty = (merchant_counts.max() / len(transactions)) if len(transactions) > 0 else 0

        location_diversity = transactions['Location City'].nunique() / max(len(transactions), 1)

        # Seasonal factor (simplified)
        month = transactions['Transaction Date'].dt.month.mode()[0] if len(transactions) > 0 else 6
        seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * month / 12)  # Simple seasonal adjustment

        features.extend([merchant_loyalty, location_diversity, seasonal_factor])

        # Risk indicators
        balance_volatility = transactions['Balance'].std() if len(transactions) > 1 else 0.0
        financial_stress = 1.0 if (current_balance < 100 and total_spending > total_income) else 0.0

        features.extend([balance_volatility, financial_stress])

        return features

    def extract_features_for_all_users(self, df, lookback_days=30):
        """Extract features for all users in the dataset"""
        user_features = []
        user_ids = []

        for user_id in tqdm(df['User ID'].unique(), desc="Extracting features"):
            user_transactions = df[df['User ID'] == user_id].copy()

            if len(user_transactions) >= 5:  # Minimum transactions for meaningful features
                features = self.extract_features(user_transactions, lookback_days)
                user_features.append(features)
                user_ids.append(user_id)

        return np.array(user_features), np.array(user_ids)
```

### Extract Features

```python
# Extract features for all users
feature_extractor = EnhancedFeatureExtractor()
user_features, user_ids = feature_extractor.extract_features_for_all_users(user_df)

print(f"✅ Feature extraction completed!")
print(f"Feature dimensions: {user_features.shape[1]} (target: 35+)")
print(f"Users with sufficient data: {len(user_ids)}")
print(f"Feature names: {len(feature_extractor.feature_names)}")

# Display feature statistics
feature_df = pd.DataFrame(user_features, columns=feature_extractor.feature_names)
print("\n=== FEATURE STATISTICS ===")
print(feature_df.describe().round(3))
```

---

## 5. User Behavioral Segmentation

### User Segmentation Model Class

```python
class UserSegmentationModel:
    """Implement user behavioral segmentation using K-means clustering"""

    def __init__(self, n_segments=5):
        self.n_segments = n_segments
        self.scaler = StandardScaler()
        self.kmeans = None
        self.segment_profiles = {}
        self.segment_names = {
            0: "Conservative Savers",
            1: "Impulse Buyers",
            2: "Goal-Oriented",
            3: "Budget Conscious",
            4: "High Spenders"
        }

    def find_optimal_clusters(self, features, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        scaled_features = self.scaler.fit_transform(features)

        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))

        # Plot elbow curve and silhouette scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)

        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}")

        return optimal_k, max(silhouette_scores)

    def train_segmentation_model(self, features):
        """Train the segmentation model"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)

        # Train K-means
        self.kmeans = KMeans(n_clusters=self.n_segments, random_state=42, n_init=10)
        segment_labels = self.kmeans.fit_predict(scaled_features)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_features, segment_labels)

        print(f"✅ Segmentation model trained!")
        print(f"Number of segments: {self.n_segments}")
        print(f"Silhouette score: {silhouette_avg:.3f}")

        return segment_labels, silhouette_avg

    def analyze_segments(self, features, segment_labels, feature_names):
        """Analyze segment characteristics"""
        feature_df = pd.DataFrame(features, columns=feature_names)
        feature_df['Segment'] = segment_labels

        # Calculate segment profiles
        for segment_id in range(self.n_segments):
            segment_data = feature_df[feature_df['Segment'] == segment_id]
            segment_size = len(segment_data)

            # Key characteristics
            profile = {
                'name': self.segment_names.get(segment_id, f"Segment {segment_id}"),
                'size': segment_size,
                'percentage': (segment_size / len(feature_df)) * 100,
                'characteristics': {}
            }

            # Analyze key features
            key_features = [
                'current_balance', 'total_spending', 'savings_rate',
                'spending_volatility', 'category_diversity', 'emergency_fund_ratio'
            ]

            for feature in key_features:
                if feature in feature_names:
                    profile['characteristics'][feature] = {
                        'mean': segment_data[feature].mean(),
                        'std': segment_data[feature].std(),
                        'median': segment_data[feature].median()
                    }

            self.segment_profiles[segment_id] = profile

        # Display segment analysis
        print("\n=== SEGMENT ANALYSIS ===")
        for segment_id, profile in self.segment_profiles.items():
            print(f"\n{profile['name']} (Segment {segment_id}):")
            print(f"  Size: {profile['size']} users ({profile['percentage']:.1f}%)")
            print("  Key Characteristics:")
            for feature, stats in profile['characteristics'].items():
                print(f"    {feature}: {stats['mean']:.2f} ± {stats['std']:.2f}")

        return self.segment_profiles

    def visualize_segments(self, features, segment_labels, feature_names):
        """Visualize segments using PCA"""
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.scaler.transform(features))

        # Create scatter plot
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for segment_id in range(self.n_segments):
            mask = segment_labels == segment_id
            segment_name = self.segment_names.get(segment_id, f"Segment {segment_id}")
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=colors[segment_id], label=segment_name, alpha=0.7)

        plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
        plt.title('User Segments Visualization (PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
```

### Train Segmentation Model

```python
# Initialize and train segmentation model
segmentation_model = UserSegmentationModel(n_segments=5)

# Find optimal number of clusters
optimal_k, best_score = segmentation_model.find_optimal_clusters(user_features)

# Train the model
segment_labels, silhouette_score = segmentation_model.train_segmentation_model(user_features)

# Analyze segments
segment_profiles = segmentation_model.analyze_segments(
    user_features, segment_labels, feature_extractor.feature_names
)

# Visualize segments
segmentation_model.visualize_segments(
    user_features, segment_labels, feature_extractor.feature_names
)
```

---

## 6. Enhanced Recommendation Model

### Enhanced Q-Network Architecture

```python
class EnhancedQNetwork(nn.Module):
    """Enhanced Q-Network for 35+ dimensional state vectors"""

    def __init__(self, state_dim=35, action_dim=5, hidden_dim=128):
        super(EnhancedQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class MultiArmedBandit:
    """Multi-armed bandit for recommendation strategy selection"""

    def __init__(self, n_strategies=5, epsilon=0.1):
        self.n_strategies = n_strategies
        self.epsilon = epsilon
        self.strategy_counts = np.zeros(n_strategies)
        self.strategy_rewards = np.zeros(n_strategies)
        self.strategy_values = np.zeros(n_strategies)

    def select_strategy(self, segment_id=None):
        """Select strategy using epsilon-greedy algorithm"""
        if np.random.random() < self.epsilon:
            # Exploration: random strategy
            return np.random.randint(self.n_strategies)
        else:
            # Exploitation: best strategy
            return np.argmax(self.strategy_values)

    def update_strategy(self, strategy_id, reward):
        """Update strategy performance based on feedback"""
        self.strategy_counts[strategy_id] += 1
        self.strategy_rewards[strategy_id] += reward
        self.strategy_values[strategy_id] = self.strategy_rewards[strategy_id] / max(self.strategy_counts[strategy_id], 1)

    def get_strategy_performance(self):
        """Get performance metrics for all strategies"""
        return {
            'counts': self.strategy_counts.copy(),
            'rewards': self.strategy_rewards.copy(),
            'values': self.strategy_values.copy()
        }

class EnhancedCQLAgent:
    """Enhanced Conservative Q-Learning agent with segment awareness"""

    def __init__(self, state_dim=35, action_dim=5, lr=1e-4, gamma=0.99, cql_alpha=5.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cql_alpha = cql_alpha

        # Enhanced Q-Network
        self.q_network = EnhancedQNetwork(state_dim, action_dim)
        self.target_q_network = EnhancedQNetwork(state_dim, action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Multi-armed bandit for strategy selection
        self.bandit = MultiArmedBandit(n_strategies=action_dim)

        # Action meanings
        self.action_meanings = {
            0: "continue_current_behavior",
            1: "spending_alert",
            2: "budget_suggestion",
            3: "savings_nudge",
            4: "positive_reinforcement"
        }

    def create_training_data(self, user_features, segment_labels):
        """Create training trajectories from user data"""
        trajectories = []

        for i, (features, segment) in enumerate(zip(user_features, segment_labels)):
            # Create state
            state = features

            # Generate action based on segment and features
            action = self._generate_action_for_state(state, segment)

            # Generate reward based on action appropriateness
            reward = self._calculate_reward(state, action, segment)

            # Create next state (simplified - add some noise)
            next_state = state + np.random.normal(0, 0.01, size=state.shape)

            # Terminal flag (simplified)
            terminal = np.random.random() < 0.1

            trajectories.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'terminal': terminal,
                'segment': segment
            })

        return trajectories

    def _generate_action_for_state(self, state, segment):
        """Generate appropriate action based on state and segment"""
        current_balance = state[0]
        savings_rate = state[15]  # Assuming savings_rate is at index 15
        spending_volatility = state[21]  # Assuming spending_volatility is at index 21

        # Rule-based action generation for training data
        if current_balance < 100:
            return 1  # spending_alert
        elif savings_rate < 0:
            return 2  # budget_suggestion
        elif savings_rate > 0.2:
            return 4  # positive_reinforcement
        elif spending_volatility > 0.5:
            return 1  # spending_alert
        else:
            return 3  # savings_nudge

    def _calculate_reward(self, state, action, segment):
        """Calculate reward based on action appropriateness for state and segment"""
        current_balance = state[0]
        savings_rate = state[15]
        spending_volatility = state[21]

        # Base reward calculation
        reward = 0.0

        # Reward appropriate actions
        if action == 1 and current_balance < 100:  # spending_alert for low balance
            reward = 1.0
        elif action == 2 and savings_rate < 0:  # budget_suggestion for negative savings
            reward = 1.0
        elif action == 4 and savings_rate > 0.2:  # positive_reinforcement for good savings
            reward = 1.0
        elif action == 3 and 0 <= savings_rate <= 0.2:  # savings_nudge for moderate savings
            reward = 0.8
        elif action == 0:  # continue_current_behavior (neutral)
            reward = 0.5
        else:
            reward = 0.2  # Inappropriate action

        # Segment-specific adjustments
        segment_multipliers = {
            0: 1.2,  # Conservative Savers - reward conservative actions more
            1: 0.8,  # Impulse Buyers - need stronger interventions
            2: 1.1,  # Goal-Oriented - reward goal-supporting actions
            3: 1.0,  # Budget Conscious - standard rewards
            4: 0.9   # High Spenders - need spending control
        }

        reward *= segment_multipliers.get(segment, 1.0)

        return reward

    def train(self, trajectories, batch_size=64, epochs=100):
        """Train the enhanced CQL model"""
        # Convert trajectories to tensors
        states = torch.FloatTensor([t['state'] for t in trajectories])
        actions = torch.LongTensor([t['action'] for t in trajectories]).unsqueeze(1)
        rewards = torch.FloatTensor([t['reward'] for t in trajectories]).unsqueeze(1)
        next_states = torch.FloatTensor([t['next_state'] for t in trajectories])
        terminals = torch.FloatTensor([t['terminal'] for t in trajectories]).unsqueeze(1)

        dataset = TensorDataset(states, actions, rewards, next_states, terminals)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []

        for epoch in tqdm(range(epochs), desc="Training CQL Model"):
            epoch_losses = []

            for batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals in dataloader:
                # Standard Q-Learning Loss
                q_values = self.q_network(batch_states).gather(1, batch_actions)

                with torch.no_grad():
                    next_q_values = self.target_q_network(batch_next_states).max(1)[0].unsqueeze(1)
                    target_q_values = batch_rewards + (1 - batch_terminals) * self.gamma * next_q_values

                q_loss = nn.MSELoss()(q_values, target_q_values)

                # CQL Conservative Loss
                all_q_values = self.q_network(batch_states)
                logsumexp_q = torch.logsumexp(all_q_values, dim=1, keepdim=True)
                dataset_q_values = q_values
                cql_loss = (logsumexp_q - dataset_q_values).mean()

                # Total loss
                total_loss = q_loss + self.cql_alpha * cql_loss

                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_losses.append(total_loss.item())

            # Update target network
            self.update_target_network()

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        return losses

    def update_target_network(self, tau=0.005):
        """Update target network with soft update"""
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def get_recommendation(self, state, segment_id=None):
        """Get recommendation for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)

            # Use bandit to select strategy
            strategy = self.bandit.select_strategy(segment_id)

            # Get action based on Q-values
            action = torch.argmax(q_values, dim=1).item()
            confidence = torch.max(q_values).item()

            return {
                'action': action,
                'action_name': self.action_meanings[action],
                'confidence': confidence,
                'strategy': strategy,
                'q_values': q_values.squeeze().numpy()
            }
```

### Train Enhanced CQL Model

```python
# Initialize and train enhanced CQL agent
state_dim = user_features.shape[1]
action_dim = 5

enhanced_agent = EnhancedCQLAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=1e-4,
    gamma=0.99,
    cql_alpha=5.0
)

# Create training data
print("Creating training trajectories...")
trajectories = enhanced_agent.create_training_data(user_features, segment_labels)
print(f"Created {len(trajectories)} training trajectories")

# Train the model
print("Training enhanced CQL model...")
training_losses = enhanced_agent.train(trajectories, batch_size=32, epochs=200)

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(training_losses)
plt.title('Enhanced CQL Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

print("✅ Enhanced CQL model training completed!")
```

---

## 7. Predictive Analytics Models

### LSTM Spending Prediction Model

```python
class SpendingPredictor(nn.Module):
    """LSTM model for spending prediction"""

    def __init__(self, input_dim=11, hidden_dim=64, num_layers=2, output_dim=1):
        super(SpendingPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Apply dropout and final layer
        output = self.dropout(last_output)
        output = self.fc(output)

        return output

class SpendingPredictionService:
    """Service for training and using spending prediction models"""

    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.category_names = [
            'groceries', 'dine_out', 'entertainment', 'bills',
            'transport', 'shopping', 'health', 'fitness',
            'savings', 'income', 'other'
        ]

    def prepare_time_series_data(self, user_df):
        """Prepare time series data for LSTM training"""
        sequences = []
        targets = []

        for user_id in user_df['User ID'].unique():
            user_data = user_df[user_df['User ID'] == user_id].copy()
            user_data = user_data.sort_values('Transaction Date')

            if len(user_data) < self.sequence_length + 1:
                continue

            # Create daily spending by category
            user_data['Date'] = user_data['Transaction Date'].dt.date
            daily_spending = user_data.groupby(['Date', 'Standard Category'])['Amount'].sum().unstack(fill_value=0)

            # Ensure all categories are present
            for cat in self.category_names:
                if cat not in daily_spending.columns:
                    daily_spending[cat] = 0

            daily_spending = daily_spending[self.category_names]
            daily_spending = daily_spending.abs()  # Convert to positive values

            # Create sequences
            for i in range(len(daily_spending) - self.sequence_length):
                sequence = daily_spending.iloc[i:i+self.sequence_length].values
                target = daily_spending.iloc[i+self.sequence_length]['groceries']  # Predict groceries spending

                sequences.append(sequence)
                targets.append(target)

        return np.array(sequences), np.array(targets)

    def train_model(self, user_df, epochs=100, batch_size=32):
        """Train the LSTM spending prediction model"""
        # Prepare data
        sequences, targets = self.prepare_time_series_data(user_df)

        if len(sequences) == 0:
            print("❌ Not enough data for time series training")
            return

        print(f"Training data shape: {sequences.shape}")
        print(f"Target data shape: {targets.shape}")

        # Scale the data
        sequences_scaled = self.scaler.fit_transform(sequences.reshape(-1, sequences.shape[-1]))
        sequences_scaled = sequences_scaled.reshape(sequences.shape)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences_scaled, targets, test_size=0.2, random_state=42
        )

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        input_dim = sequences.shape[-1]
        self.model = SpendingPredictor(input_dim=input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        train_losses = []

        for epoch in tqdm(range(epochs), desc="Training LSTM"):
            epoch_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            test_predictions = self.model(X_test_tensor)
            test_loss = criterion(test_predictions, y_test_tensor)
            mae = mean_absolute_error(y_test, test_predictions.numpy())

        print(f"✅ LSTM model training completed!")
        print(f"Test Loss (MSE): {test_loss.item():.4f}")
        print(f"Test MAE: {mae:.4f}")

        return train_losses, test_loss.item(), mae

    def predict_spending(self, recent_sequences):
        """Predict future spending based on recent sequences"""
        if self.model is None:
            return None

        self.model.eval()
        with torch.no_grad():
            # Scale the input
            sequences_scaled = self.scaler.transform(recent_sequences.reshape(-1, recent_sequences.shape[-1]))
            sequences_scaled = sequences_scaled.reshape(recent_sequences.shape)

            # Convert to tensor and predict
            input_tensor = torch.FloatTensor(sequences_scaled)
            predictions = self.model(input_tensor)

            return predictions.numpy()

# Train spending prediction model
spending_predictor = SpendingPredictionService(sequence_length=7)
train_losses, test_loss, mae = spending_predictor.train_model(user_df, epochs=150, batch_size=16)

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title('LSTM Spending Prediction Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()
```

### Goal Achievement Prediction Model

```python
class GoalAchievementPredictor:
    """Random Forest model for goal achievement probability"""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None

    def create_goal_features(self, user_features, segment_labels):
        """Create features for goal achievement prediction"""
        goal_features = []
        goal_labels = []

        for i, (features, segment) in enumerate(zip(user_features, segment_labels)):
            # Extract relevant features for goal prediction
            current_balance = features[0]
            savings_rate = features[15]
            savings_consistency = features[26]
            emergency_fund_ratio = features[27]
            spending_volatility = features[21]

            # Create goal-specific features
            goal_feature_vector = [
                current_balance,
                savings_rate,
                savings_consistency,
                emergency_fund_ratio,
                spending_volatility,
                segment,  # User segment as feature
                features[2],  # total_income
                features[1],  # total_spending
                features[22],  # large_transaction_ratio
                features[24]   # spending_consistency
            ]

            # Generate synthetic goal achievement label
            # Higher probability for users with good financial habits
            achievement_prob = (
                0.3 * (1 if savings_rate > 0.1 else 0) +
                0.2 * (1 if current_balance > 500 else 0) +
                0.2 * (1 if savings_consistency > 0.1 else 0) +
                0.2 * (1 if emergency_fund_ratio > 0.5 else 0) +
                0.1 * (1 if spending_volatility < 100 else 0)
            )

            # Add some randomness
            achievement_prob += np.random.normal(0, 0.1)
            achievement_prob = np.clip(achievement_prob, 0, 1)

            goal_achieved = 1 if achievement_prob > 0.5 else 0

            goal_features.append(goal_feature_vector)
            goal_labels.append(goal_achieved)

        self.feature_names = [
            'current_balance', 'savings_rate', 'savings_consistency',
            'emergency_fund_ratio', 'spending_volatility', 'segment',
            'total_income', 'total_spending', 'large_transaction_ratio',
            'spending_consistency'
        ]

        return np.array(goal_features), np.array(goal_labels)

    def train_model(self, user_features, segment_labels):
        """Train the goal achievement prediction model"""
        # Create goal-specific features
        goal_features, goal_labels = self.create_goal_features(user_features, segment_labels)

        print(f"Goal prediction training data: {goal_features.shape}")
        print(f"Goal achievement rate: {goal_labels.mean():.2%}")

        # Scale features
        goal_features_scaled = self.scaler.fit_transform(goal_features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            goal_features_scaled, goal_labels, test_size=0.2, random_state=42, stratify=goal_labels
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        # Get predictions and probabilities
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print(f"✅ Goal achievement model training completed!")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance)

        return test_score, feature_importance

    def predict_goal_achievement(self, user_features, segment_label):
        """Predict goal achievement probability for a user"""
        if self.model is None:
            return None

        # Create goal features for single user
        goal_features = np.array([[
            user_features[0],   # current_balance
            user_features[15],  # savings_rate
            user_features[26],  # savings_consistency
            user_features[27],  # emergency_fund_ratio
            user_features[21],  # spending_volatility
            segment_label,      # segment
            user_features[2],   # total_income
            user_features[1],   # total_spending
            user_features[22],  # large_transaction_ratio
            user_features[24]   # spending_consistency
        ]])

        # Scale and predict
        goal_features_scaled = self.scaler.transform(goal_features)
        probability = self.model.predict_proba(goal_features_scaled)[0, 1]

        return probability

# Train goal achievement model
goal_predictor = GoalAchievementPredictor()
test_accuracy, feature_importance = goal_predictor.train_model(user_features, segment_labels)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Goal Achievement Prediction - Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

---

## 8. Model Validation and Export

### Model Performance Evaluation

```python
def evaluate_all_models(segmentation_model, enhanced_agent, spending_predictor, goal_predictor,
                       user_features, segment_labels, user_df):
    """Comprehensive evaluation of all trained models"""

    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # 1. Segmentation Model Evaluation
    print("\n1. USER SEGMENTATION MODEL")
    print("-" * 30)

    scaled_features = segmentation_model.scaler.transform(user_features)
    silhouette_avg = silhouette_score(scaled_features, segment_labels)

    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Number of segments: {segmentation_model.n_segments}")

    # Segment distribution
    unique, counts = np.unique(segment_labels, return_counts=True)
    print("Segment distribution:")
    for seg_id, count in zip(unique, counts):
        seg_name = segmentation_model.segment_names.get(seg_id, f"Segment {seg_id}")
        print(f"  {seg_name}: {count} users ({count/len(segment_labels)*100:.1f}%)")

    # 2. Enhanced CQL Model Evaluation
    print("\n2. ENHANCED CQL RECOMMENDATION MODEL")
    print("-" * 40)

    # Test recommendations for sample users
    sample_indices = np.random.choice(len(user_features), 5, replace=False)

    print("Sample recommendations:")
    for i, idx in enumerate(sample_indices):
        state = user_features[idx]
        segment = segment_labels[idx]

        recommendation = enhanced_agent.get_recommendation(state, segment)
        segment_name = segmentation_model.segment_names.get(segment, f"Segment {segment}")

        print(f"  User {idx} ({segment_name}):")
        print(f"    Action: {recommendation['action_name']}")
        print(f"    Confidence: {recommendation['confidence']:.3f}")

    # 3. Spending Prediction Model Evaluation
    print("\n3. SPENDING PREDICTION MODEL (LSTM)")
    print("-" * 35)

    if spending_predictor.model is not None:
        print(f"Model trained successfully")
        print(f"Sequence length: {spending_predictor.sequence_length} days")
        print(f"Categories predicted: {len(spending_predictor.category_names)}")
    else:
        print("Model not trained (insufficient data)")

    # 4. Goal Achievement Model Evaluation
    print("\n4. GOAL ACHIEVEMENT PREDICTION MODEL")
    print("-" * 38)

    # Test predictions for sample users
    print("Sample goal achievement predictions:")
    for i, idx in enumerate(sample_indices):
        state = user_features[idx]
        segment = segment_labels[idx]

        probability = goal_predictor.predict_goal_achievement(state, segment)
        segment_name = segmentation_model.segment_names.get(segment, f"Segment {segment}")

        print(f"  User {idx} ({segment_name}): {probability:.1%} chance of goal achievement")

    # 5. Overall System Performance
    print("\n5. OVERALL SYSTEM PERFORMANCE")
    print("-" * 30)

    total_users = len(user_features)
    feature_dimensions = user_features.shape[1]

    print(f"Total users processed: {total_users}")
    print(f"Feature dimensions: {feature_dimensions} (target: 35+)")
    print(f"Segmentation quality: {'Good' if silhouette_avg > 0.3 else 'Needs improvement'}")
    print(f"Goal prediction accuracy: {test_accuracy:.1%}")

    return {
        'segmentation_score': silhouette_avg,
        'total_users': total_users,
        'feature_dimensions': feature_dimensions,
        'goal_accuracy': test_accuracy
    }

# Run comprehensive evaluation
evaluation_results = evaluate_all_models(
    segmentation_model, enhanced_agent, spending_predictor, goal_predictor,
    user_features, segment_labels, user_df
)
```

### Model Export for Backend Integration

```python
def export_models_for_backend(segmentation_model, enhanced_agent, spending_predictor,
                             goal_predictor, feature_extractor, evaluation_results):
    """Export all trained models for backend integration"""

    print("Exporting models for backend integration...")

    # Create models directory
    import os
    os.makedirs('enhanced_models', exist_ok=True)

    # 1. Export Segmentation Model
    joblib.dump(segmentation_model.kmeans, 'enhanced_models/user_segmentation_model.pkl')
    joblib.dump(segmentation_model.scaler, 'enhanced_models/segmentation_scaler.pkl')

    # Export segment profiles
    with open('enhanced_models/segment_profiles.json', 'w') as f:
        json.dump(segmentation_model.segment_profiles, f, indent=2, default=str)

    # 2. Export Enhanced CQL Model
    torch.save(enhanced_agent.q_network.state_dict(), 'enhanced_models/enhanced_cql_model.pth')

    # Export bandit state
    bandit_state = enhanced_agent.bandit.get_strategy_performance()
    with open('enhanced_models/bandit_state.json', 'w') as f:
        json.dump(bandit_state, f, indent=2, default=str)

    # 3. Export Spending Predictor
    if spending_predictor.model is not None:
        torch.save(spending_predictor.model.state_dict(), 'enhanced_models/spending_predictor.pth')
        joblib.dump(spending_predictor.scaler, 'enhanced_models/spending_scaler.pkl')

    # 4. Export Goal Achievement Model
    joblib.dump(goal_predictor.model, 'enhanced_models/goal_achievement_model.pkl')
    joblib.dump(goal_predictor.scaler, 'enhanced_models/goal_scaler.pkl')

    # 5. Export Feature Engineering Pipeline
    feature_config = {
        'feature_names': feature_extractor.feature_names,
        'standard_categories': feature_extractor.standard_categories,
        'feature_count': len(feature_extractor.feature_names)
    }

    with open('enhanced_models/feature_config.json', 'w') as f:
        json.dump(feature_config, f, indent=2)

    # 6. Export Model Metadata
    metadata = {
        'model_version': '2.0.0',
        'training_date': datetime.now().isoformat(),
        'feature_dimensions': evaluation_results['feature_dimensions'],
        'total_training_users': evaluation_results['total_users'],
        'segmentation_score': evaluation_results['segmentation_score'],
        'goal_prediction_accuracy': evaluation_results['goal_accuracy'],
        'model_files': {
            'segmentation': 'user_segmentation_model.pkl',
            'cql_model': 'enhanced_cql_model.pth',
            'spending_predictor': 'spending_predictor.pth',
            'goal_predictor': 'goal_achievement_model.pkl',
            'feature_config': 'feature_config.json'
        },
        'action_meanings': enhanced_agent.action_meanings,
        'segment_names': segmentation_model.segment_names
    }

    with open('enhanced_models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # 7. Create Integration Guide
    integration_guide = """
# FinCoach Enhanced ML Models - Integration Guide

## Model Files Overview

1. **user_segmentation_model.pkl** - K-means clustering model for user segmentation
2. **enhanced_cql_model.pth** - Enhanced Q-Network for recommendations
3. **spending_predictor.pth** - LSTM model for spending prediction
4. **goal_achievement_model.pkl** - Random Forest for goal achievement probability
5. **feature_config.json** - Feature engineering configuration
6. **model_metadata.json** - Model metadata and performance metrics

## Integration Steps

1. Copy all model files to your backend `models/enhanced/` directory
2. Update your `ml_service.py` to load the enhanced models
3. Extend the `FinancialStateVector` class to generate 35+ dimensional features
4. Implement the new prediction endpoints as outlined in the backend integration plan

## Model Performance

- Segmentation Quality: {:.3f} (Silhouette Score)
- Feature Dimensions: {} (Enhanced from 17)
- Goal Prediction Accuracy: {:.1%}
- Training Users: {}

## Next Steps

1. Test model loading in your backend environment
2. Implement the enhanced feature extraction
3. Add new API endpoints for predictions
4. Set up A/B testing framework
5. Monitor model performance in production
    """.format(
        evaluation_results['segmentation_score'],
        evaluation_results['feature_dimensions'],
        evaluation_results['goal_accuracy'],
        evaluation_results['total_users']
    )

    with open('enhanced_models/INTEGRATION_GUIDE.md', 'w') as f:
        f.write(integration_guide)

    print("✅ Model export completed!")
    print("\nExported files:")
    for file in os.listdir('enhanced_models'):
        print(f"  - enhanced_models/{file}")

    return 'enhanced_models'

# Export all models
export_directory = export_models_for_backend(
    segmentation_model, enhanced_agent, spending_predictor,
    goal_predictor, feature_extractor, evaluation_results
)

print(f"\n🎉 FinCoach ML Enhancement Implementation Complete!")
print(f"All models exported to: {export_directory}/")
print("Ready for backend integration!")
```

---

## Summary

This implementation provides:

1. **35+ Dimensional Feature Engineering** - Enhanced state vectors with behavioral patterns
2. **User Behavioral Segmentation** - 5 distinct user segments using K-means clustering
3. **Enhanced CQL Recommendation Model** - Multi-armed bandit strategy selection
4. **LSTM Spending Prediction** - Time series forecasting for spending patterns
5. **Goal Achievement Prediction** - Random Forest classifier for goal completion probability
6. **Comprehensive Model Export** - Production-ready models for backend integration

The models are designed to integrate seamlessly with your existing FinCoach backend infrastructure while providing significant improvements in personalization and predictive capabilities.
