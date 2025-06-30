"""
Database models and operations for Rice GrainPalette application
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
import json

# Database configuration
DATABASE_URL = os.getenv('C:\RiceTypeDetection\__pycache__\database.cpython-312.pyc')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionHistory(Base):
    """Store rice classification prediction history"""
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    all_predictions = Column(Text)  # JSON string of all class predictions
    image_size = Column(String(20))  # e.g., "224x224"
    image_format = Column(String(10))  # e.g., "JPEG", "PNG"
    model_accuracy = Column(Float)
    processing_time_ms = Column(Float)
    
    def __repr__(self):
        return f"<PredictionHistory(id={self.id}, class='{self.predicted_class}', confidence={self.confidence})>"

class ModelMetrics(Base):
    """Store model performance metrics and statistics"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_version = Column(String(20), default="1.0")
    accuracy = Column(Float)
    total_predictions = Column(Integer, default=0)
    predictions_today = Column(Integer, default=0)
    most_predicted_class = Column(String(50))
    least_predicted_class = Column(String(50))
    average_confidence = Column(Float)
    
    def __repr__(self):
        return f"<ModelMetrics(id={self.id}, accuracy={self.accuracy}, total={self.total_predictions})>"

class UserSession(Base):
    """Track user sessions and usage patterns"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_start = Column(DateTime, default=datetime.utcnow, index=True)
    session_end = Column(DateTime)
    predictions_count = Column(Integer, default=0)
    unique_classes_predicted = Column(Integer, default=0)
    session_duration_minutes = Column(Float)
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, predictions={self.predictions_count})>"

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ Database tables created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create database tables: {e}")
        return False

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_prediction(predicted_class, confidence, all_predictions, image_size=None, 
                  image_format=None, model_accuracy=None, processing_time_ms=None):
    """Add a new prediction to the database"""
    db = SessionLocal()
    try:
        prediction = PredictionHistory(
            predicted_class=predicted_class,
            confidence=confidence,
            all_predictions=json.dumps(all_predictions),
            image_size=image_size,
            image_format=image_format,
            model_accuracy=model_accuracy,
            processing_time_ms=processing_time_ms
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction.id
    except Exception as e:
        db.rollback()
        print(f"Error adding prediction: {e}")
        return None
    finally:
        db.close()

def get_prediction_stats():
    """Get prediction statistics"""
    db = SessionLocal()
    try:
        total_predictions = db.query(PredictionHistory).count()
        
        if total_predictions == 0:
            return {
                'total_predictions': 0,
                'average_confidence': 0,
                'most_predicted_class': None,
                'class_distribution': {}
            }
        
        # Get class distribution
        from sqlalchemy import func
        class_counts = db.query(
            PredictionHistory.predicted_class,
            func.count(PredictionHistory.predicted_class).label('count')
        ).group_by(PredictionHistory.predicted_class).all()
        
        class_distribution = {cls: count for cls, count in class_counts}
        most_predicted_class = max(class_distribution.items(), key=lambda x: x[1])[0] if class_distribution else None
        
        # Get average confidence
        avg_confidence = db.query(func.avg(PredictionHistory.confidence)).scalar() or 0
        
        return {
            'total_predictions': total_predictions,
            'average_confidence': float(avg_confidence),
            'most_predicted_class': most_predicted_class,
            'class_distribution': class_distribution
        }
        
    except Exception as e:
        print(f"Error getting prediction stats: {e}")
        return {
            'total_predictions': 0,
            'average_confidence': 0,
            'most_predicted_class': None,
            'class_distribution': {}
        }
    finally:
        db.close()

def get_recent_predictions(limit=10):
    """Get recent predictions"""
    db = SessionLocal()
    try:
        predictions = db.query(PredictionHistory)\
                       .order_by(PredictionHistory.timestamp.desc())\
                       .limit(limit)\
                       .all()
        
        result = []
        for pred in predictions:
            result.append({
                'id': pred.id,
                'timestamp': pred.timestamp.isoformat(),
                'predicted_class': pred.predicted_class,
                'confidence': pred.confidence,
                'all_predictions': json.loads(pred.all_predictions) if pred.all_predictions and isinstance(pred.all_predictions, str) else [],
                'image_size': pred.image_size,
                'image_format': pred.image_format,
                'processing_time_ms': pred.processing_time_ms
            })
        
        return result
        
    except Exception as e:
        print(f"Error getting recent predictions: {e}")
        return []
    finally:
        db.close()

def update_model_metrics(accuracy, total_predictions):
    """Update model metrics"""
    db = SessionLocal()
    try:
        # Get today's date for filtering
        today = datetime.utcnow().date()
        
        # Count today's predictions
        from sqlalchemy import func, cast, Date
        predictions_today = db.query(PredictionHistory)\
                             .filter(cast(PredictionHistory.timestamp, Date) == today)\
                             .count()
        
        # Get statistics
        stats = get_prediction_stats()
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            total_predictions=total_predictions,
            predictions_today=predictions_today,
            most_predicted_class=stats.get('most_predicted_class'),
            average_confidence=stats.get('average_confidence')
        )
        
        db.add(metrics)
        db.commit()
        return True
        
    except Exception as e:
        db.rollback()
        print(f"Error updating model metrics: {e}")
        return False
    finally:
        db.close()

def check_database_connection():
    """Check if database connection is working"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def initialize_database():
    """Initialize the database with tables and basic setup"""
    print("Initializing database...")
    
    # Check connection
    if not check_database_connection():
        print("✗ Database connection failed")
        return False
    
    # Create tables
    if not create_tables():
        return False
    
    print("✓ Database initialized successfully")
    return True

if __name__ == "__main__":
    # Test database setup
    if initialize_database():
        print("\nTesting database operations...")
        
        # Test adding a prediction
        pred_id = add_prediction(
            predicted_class="Basmati",
            confidence=0.92,
            all_predictions=[
                {"class": "Basmati", "confidence": 0.92},
                {"class": "Jasmine", "confidence": 0.05},
                {"class": "Arborio", "confidence": 0.02},
                {"class": "Ipsala", "confidence": 0.01},
                {"class": "Karacadag", "confidence": 0.00}
            ],
            image_size="224x224",
            image_format="JPEG",
            model_accuracy=0.892,
            processing_time_ms=150.5
        )
        
        if pred_id:
            print(f"✓ Test prediction added with ID: {pred_id}")
            
            # Test getting stats
            stats = get_prediction_stats()
            print(f"✓ Database stats: {stats}")
            
            # Test getting recent predictions
            recent = get_recent_predictions(5)
            print(f"✓ Recent predictions count: {len(recent)}")
            
        else:
            print("✗ Failed to add test prediction")
    else:
        print("✗ Database initialization failed")