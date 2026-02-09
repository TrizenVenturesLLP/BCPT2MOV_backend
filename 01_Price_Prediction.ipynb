{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Market Price Prediction\n",
    "\n",
    "This notebook focuses on predicting the market price (`selling_price_per_kg`) of crops based on various features such as crop type, location, quantity, and market conditions.\n",
    "\n",
    "## Objectives\n",
    "1. Load and explore the dataset.\n",
    "2. List all available Market Access locations.\n",
    "3. Preprocess the data (encoding, scaling).\n",
    "4. Train a Machine Learning model (Random Forest).\n",
    "5. Evaluate the model performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "# Set plot style\n",
    "sns.set(style=\"whitegrid\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>commodity_group</th>\n",
       "      <th>crop_type</th>\n",
       "      <th>state_name</th>\n",
       "      <th>market_location</th>\n",
       "      <th>quantity_kg</th>\n",
       "      <th>quality_grade</th>\n",
       "      <th>season</th>\n",
       "      <th>transport_cost</th>\n",
       "      <th>demand_index</th>\n",
       "      <th>selling_price_per_kg</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2023-01-01</td>\n",
       "      <td>Millet</td>\n",
       "      <td>Jowar (Sorghum)</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Banaganapalli</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Rabi</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.10</td>\n",
       "      <td>29.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2023-01-01</td>\n",
       "      <td>Millet</td>\n",
       "      <td>Jowar (Sorghum)</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Alur</td>\n",
       "      <td>10.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Rabi</td>\n",
       "      <td>4.94</td>\n",
       "      <td>0.10</td>\n",
       "      <td>28.60</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2023-01-02</td>\n",
       "      <td>Oilseed</td>\n",
       "      <td>Sunflower</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Kurnool</td>\n",
       "      <td>2300.0</td>\n",
       "      <td>2</td>\n",
       "      <td>Rabi</td>\n",
       "      <td>3.14</td>\n",
       "      <td>0.49</td>\n",
       "      <td>61.89</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2023-01-02</td>\n",
       "      <td>Millet</td>\n",
       "      <td>Jowar (Sorghum)</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Alur</td>\n",
       "      <td>10.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Rabi</td>\n",
       "      <td>4.94</td>\n",
       "      <td>0.10</td>\n",
       "      <td>28.60</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2023-01-02</td>\n",
       "      <td>Millet</td>\n",
       "      <td>Jowar (Sorghum)</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Banaganapalli</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Rabi</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.10</td>\n",
       "      <td>29.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         date commodity_group        crop_type      state_name  \\\n",
       "0  2023-01-01          Millet  Jowar (Sorghum)  Andhra Pradesh   \n",
       "1  2023-01-01          Millet  Jowar (Sorghum)  Andhra Pradesh   \n",
       "2  2023-01-02         Oilseed        Sunflower  Andhra Pradesh   \n",
       "3  2023-01-02          Millet  Jowar (Sorghum)  Andhra Pradesh   \n",
       "4  2023-01-02          Millet  Jowar (Sorghum)  Andhra Pradesh   \n",
       "\n",
       "  market_location  quantity_kg  quality_grade season  transport_cost  \\\n",
       "0   Banaganapalli       1000.0              3   Rabi            3.26   \n",
       "1            Alur         10.0              3   Rabi            4.94   \n",
       "2         Kurnool       2300.0              2   Rabi            3.14   \n",
       "3            Alur         10.0              3   Rabi            4.94   \n",
       "4   Banaganapalli       1000.0              3   Rabi            3.26   \n",
       "\n",
       "   demand_index  selling_price_per_kg  \n",
       "0          0.10                 29.00  \n",
       "1          0.10                 28.60  \n",
       "2          0.49                 61.89  \n",
       "3          0.10                 28.60  \n",
       "4          0.10                 29.00  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load the dataset\n",
    "file_path = 'price_suggestion_dataset.csv'\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# Display first few rows\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Market Access List\n",
    "Here we list all the unique market locations present in the dataset, which represent the market access points for farmers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total Market Access Locations: 9\n",
      "List of Markets:\n",
      "- Adoni\n",
      "- Alur\n",
      "- Banaganapalli\n",
      "- Chintapally\n",
      "- Kurnool\n",
      "- Nandyal\n",
      "- Rapur\n",
      "- Tuni\n",
      "- Yemmiganur\n"
     ]
    }
   ],
   "source": [
    "market_access_list = df['market_location'].unique()\n",
    "print(f\"Total Market Access Locations: {len(market_access_list)}\")\n",
    "print(\"List of Markets:\")\n",
    "for market in sorted(market_access_list):\n",
    "    print(f\"- {market}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing & EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 3179 entries, 0 to 3178\n",
      "Data columns (total 11 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   date                  3179 non-null   object \n",
      " 1   commodity_group       3179 non-null   object \n",
      " 2   crop_type             3179 non-null   object \n",
      " 3   state_name            3179 non-null   object \n",
      " 4   market_location       3179 non-null   object \n",
      " 5   quantity_kg           3179 non-null   float64\n",
      " 6   quality_grade         3179 non-null   int64  \n",
      " 7   season                3179 non-null   object \n",
      " 8   transport_cost        3179 non-null   float64\n",
      " 9   demand_index          3179 non-null   float64\n",
      " 10  selling_price_per_kg  3179 non-null   float64\n",
      "dtypes: float64(4), int64(1), object(6)\n",
      "memory usage: 273.3+ KB\n",
      "None\n",
      "\n",
      "Missing Values:\n",
      "date                    0\n",
      "commodity_group         0\n",
      "crop_type               0\n",
      "state_name              0\n",
      "market_location         0\n",
      "quantity_kg             0\n",
      "quality_grade           0\n",
      "season                  0\n",
      "transport_cost          0\n",
      "demand_index            0\n",
      "selling_price_per_kg    0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Check info and missing values\n",
    "print(df.info())\n",
    "print(\"\\nMissing Values:\")\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert date to datetime\n",
    "df['date'] = pd.to_datetime(df['date'])\n",
    "\n",
    "# Extract temporal features\n",
    "df['month'] = df['date'].dt.month\n",
    "df['day_of_week'] = df['date'].dt.dayofweek\n",
    "\n",
    "# Drop original date column for training\n",
    "df_train = df.drop(columns=['date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>commodity_group</th>\n",
       "      <th>crop_type</th>\n",
       "      <th>state_name</th>\n",
       "      <th>market_location</th>\n",
       "      <th>quantity_kg</th>\n",
       "      <th>quality_grade</th>\n",
       "      <th>season</th>\n",
       "      <th>transport_cost</th>\n",
       "      <th>demand_index</th>\n",
       "      <th>selling_price_per_kg</th>\n",
       "      <th>month</th>\n",
       "      <th>day_of_week</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.10</td>\n",
       "      <td>29.00</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>4.94</td>\n",
       "      <td>0.10</td>\n",
       "      <td>28.60</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2300.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3.14</td>\n",
       "      <td>0.49</td>\n",
       "      <td>61.89</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>4.94</td>\n",
       "      <td>0.10</td>\n",
       "      <td>28.60</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.10</td>\n",
       "      <td>29.00</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   commodity_group  crop_type  state_name  market_location  quantity_kg  \\\n",
       "0                0          3           0                2       1000.0   \n",
       "1                0          3           0                1         10.0   \n",
       "2                1          7           0                4       2300.0   \n",
       "3                0          3           0                1         10.0   \n",
       "4                0          3           0                2       1000.0   \n",
       "\n",
       "   quality_grade  season  transport_cost  demand_index  selling_price_per_kg  \\\n",
       "0              3       1            3.26          0.10                 29.00   \n",
       "1              3       1            4.94          0.10                 28.60   \n",
       "2              2       1            3.14          0.49                 61.89   \n",
       "3              3       1            4.94          0.10                 28.60   \n",
       "4              3       1            3.26          0.10                 29.00   \n",
       "\n",
       "   month  day_of_week  \n",
       "0      1            6  \n",
       "1      1            6  \n",
       "2      1            0  \n",
       "3      1            0  \n",
       "4      1            0  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Encode Categorical Variables\n",
    "categorical_cols = ['commodity_group', 'crop_type', 'state_name', 'market_location', 'season']\n",
    "label_encoders = {}\n",
    "\n",
    "for col in categorical_cols:\n",
    "    le = LabelEncoder()\n",
    "    df_train[col] = le.fit_transform(df_train[col])\n",
    "    label_encoders[col] = le\n",
    "\n",
    "df_train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training shape: (2543, 11)\n",
      "Testing shape: (636, 11)\n"
     ]
    }
   ],
   "source": [
    "# Define Features (X) and Target (y)\n",
    "X = df_train.drop(columns=['selling_price_per_kg'])\n",
    "y = df_train['selling_price_per_kg']\n",
    "\n",
    "# Split into Train and Test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "print(f\"Training shape: {X_train.shape}\")\n",
    "print(f\"Testing shape: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train Random Forest Regressor\n",
    "rf_model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "rf_model.fit(X_train, y_train)\n",
    "\n",
    "# Predictions\n",
    "y_pred = rf_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Root Mean Squared Error (RMSE): 4.86\n",
      "R^2 Score: 0.87\n"
     ]
    }
   ],
   "source": [
    "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print(f\"Root Mean Squared Error (RMSE): {rmse:.2f}\")\n",
    "print(f\"R^2 Score: {r2:.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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6f64iMW5LmmBGoVGjUKjRuGhTqHFeo2W/ff7W9dRjvqOIgAAAKJThq96jkUa4tWQeUqU4lN7KR0AAID0RFBMB5r/OHPmzBS3OeWUUyxWFCqQJ1PsEwAARIagmA60qCT4AuCxrnvLKumy38REn8XHR+c7xgAAkBUwRxGpossJpddb7BESAQDIWARFpBoL5wEAiE0ERQAAAHgiKAIAAMATQREAAACeCIoAAADwRFAEAACAJ4IiAAAAPBEUAQAA4ImgCAAAAE8ERQAAAHgiKAIAAMATQREAAACeCIoAAADwRFAEAACAJ4IiAAAAPBEUkWpxcXEZfQgAACAdEBSRKgkJCZYrV650fY3ERF+67h8AAHjLfpz7gbANf3Wxbdu5J132XahAHuveskq67BsAAKSMoIhUU0hcv313Rh8GAABIYww9AwAAwBNBMZ398ssv9v777wdu16tXz8aMGeM+9/l89s4779iuXbvS5LXmz59vpUqVsm3btqXJ/gAAQNZGUExnjzzyiH399deB22+99Zbddddd7vOFCxdaz549bf/+/Rl4hAAAAN6Yo3iS5cuXL/C5OooAAADRKqY7ijt37rT777/fKlWqZFdccYW9+eabbuh3xowZbvhXnwdLft+aNWvsnnvusWrVqlnZsmXtqquuskmTJiXZ/s4777QJEybY5ZdfbuXKlbNWrVrZ+vXr3eOtW7e2BQsWuOFl/379Q88aJm7Tpo27T/udPn261axZ08aOHZvkmN544w2rU6eOHTlyJOLzX7RokTv3kSNHBu6bPHmyO4by5ctb27Zt3eslrwMAAEBMdxQPHz7shnhPO+00e+WVV+zAgQPWp08f++2338J6voaD9fzatWu7sJYtWzYXNIcMGeIC3cUXXxwIY6eccooLi3rNHj16WL9+/WzKlCkuEHbs2NHOPfdce+KJJ5LsXwFOj3fq1Mntt2TJki5gzp492x544IHAdjNnzrTGjRtb9uyRfal++OEH69ChgwuDnTt3dve9+uqrLjT27t3bqlSpYh999JE988wzdt5551m009cjs3Zg/VMLmGJwfNQoNGoUGjUKD3UKLdZr5PP5wn6zjJgNit9++62tW7fOPv74YytWrJi7b/DgwXbTTTeF9Xx9c6jj17JlSzv11FPdfQpcL774oq1evToQFNXpGzp0qOXNm9fdvu2222zYsGHu8zPO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11qHDh2sevXqVrNmTevcubP98ssvFssirdHPP/9sd9xxh1WqVMlq1KhhTzzxhO3Zs8di2Yn+rMns2bOtVKlStm3bNot1kdbJX5vkH7Fcq0hrdPjwYXv66acD27dq1cpWrlxpWQFBEcc1btw4e+2116x///72xhtvBILNoUOHPLfXL/PNmzfb5MmTbfTo0fbVV19Z3759LdZFWifRY48++qh9/fXXlhVEUiP9wdG2bVvLmTOnvfLKK/bCCy/Yn3/+6bY/ePCgxapIavTHH3+4GhUsWNBmzJjhnrt48WLr2bOnxbIT+VmT7du325NPPmlZRaR1Wr16tfvjft68eUk+zjvvPItVkdaob9++7mdt4MCB9vbbb1u+fPlcuIz1P84cH+Dh4MGDvkqVKvleffXVwH27d+/2lS9f3vfuu+8es/2SJUt8JUuW9K1bty5w39dff+0rVaqUb8eOHb5YFWmdZPHixb6GDRv6rrrqKl/VqlV9zzzzjC+WRVqj6dOnu+33798fuO+XX35x31/ffvutLxZFWqMffvjB99BDD/kOHz4cuG/y5Mm+ChUq+GLVifysydGjR3233367r02bNu57aOvWrb5YdiJ1ateuna9///6+rCLSGm3ZssX9Lvviiy+SbH/llVfG7P+TgtFRhKdVq1bZ3r173bCf3+mnn25lypSxhQsXHrP9okWLLH/+/FaiRInAffoLNS4uznU6YlWkdRJ1WjV8MXPmTMuTJ4/FukhrpO301746in7x8f/7X9U///xjsSjSGlWoUMFGjBhh2bNnd7fXr19vs2bNstq1a1usOpGfNRk/frwbNrznnnssKziROqmjGPz/7lgXaY2++eYb9//qyy+/PMn2n3/+eZJ9xKr//V8GSGbHjh3u3+RDD+ecc07gsWA7d+48ZtuEhAQ744wz7Ndff7VYFWmd5KGHHrKsJNIaFSpUyH0EmzBhgguOmv8ai07k+8jv2muvtU2bNrlh6LFjx1qsOpEa/fTTTzZp0iR766233P+jsoJI67R7925XG/2xr6FYTf0oX768Pfzww1a8eHGLRZHWaOPGjVa4cGH75JNP3P+LVC+FSk31yAoBm44iPPkXoSjsBTvllFM854lp++TbprR9Vq1TVpTaGmme4tSpU6179+5uXlAsSk2Nhg8f7mp01llnWZs2bVynJBZFWqN9+/a57xl9FCtWzLKKSOukhWPi8/ls0KBBNmrUKLddixYt3FzYWBRpjf799183/14jHV27drXnnnvOdfNVo127dlmsIyjCk3/YL/nEXv0Q5cqVy3N7r0nA2j537twWqyKtU1Z0ojXSLy790howYIDde++91rp1a4tVqfk+KleunJvmoW6iVqnOmTPHYlGkNdL3jTpit912m2UlkdapatWq9t1337kVvWXLlnW39b2kxR1avBGLIq1R9uzZXVgcOXKk1alTx3Vc9bm88847FusIivDkb8n/9ttvSe7X7QIFChyz/bnnnnvMtvoh/Pvvv107P1ZFWqes6ERqpDllGvrS/LJevXpZly5dLJZFWqMNGzbYl19+meQ+baepHrE6xBppjbQy9dtvv3WXD9KHVqhKo0aN3PdVrDqRnzd16jWf3E9hSdM/+F76H/1+U1gMHmZW2NRwdCxfQsiPoAhPpUuXttNOO83mz58fuE8LCVasWOE5T0z3aW6H2vN+CxYscP9WqVLFYlWkdcqKTqRGPXr0sI8++sh1Oe68806LdZHWSAFIl6MKXtyzZcsWN78sVudMRVojzSd777333KIxfajDKJpjFstdxkjrNG3aNHe9Ug3V+6l7pnmvF154ocWiE/n9duTIEVu2bFngvgMHDrjrLhYtWtRiHYtZ4ElzN3RBUc1/0l+bmig/bNgw95fVNddcY0ePHnXXttNKMP1lpVWYlStXdgs1dL0p/U9HFwBu2rRpTHfWIq1TVhRpjTTc9cEHH7iwqCHV33//PbCvWK1jpDVSV0yBR11XzcHTggQFIQ2JXXnllRaLIq1R8l/g/kUK559/vuu8xqpI66SVvNpWP28PPvigC0BaUa/nNmvWzGJRpDWqWrWq1apVyx555BF3PU59/+hi3dmyZbMmTZpYzMvo6/Mgeh05csQ3dOhQX40aNXwVK1b0tW/fPnANMv2ra5K9/fbbge3/+OMPX6dOndy21atX9/Xp08d34MABX6yLtE7BdB2uWL+OYqQ1atu2rbvt9XG8OmbF76MNGzb4OnTo4KtSpYrv0ksv9fXq1ctd2y2WpeZn7fvvv88S11E8kTotX77c/dzpe6ly5cru/+O6dmksi7RGe/bscb/T9LtN1ytVvdauXevLCuL0n4wOqwAAAIg+zFEEAACAJ4IiAAAAPBEUAQAA4ImgCAAAAE8ERQAAAHgiKAIAAMATQREAAACeCIoAAADwRFAEAACAJ4IiAAAAPBEUAQAA4ImgCAAAAPPyfzHF+7dKv0obAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Feature Importance\n",
    "feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)\n",
    "feature_importances.nlargest(10).plot(kind='barh')\n",
    "plt.title('Top 10 Feature Importances')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
