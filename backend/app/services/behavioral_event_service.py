import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from app.models.database import BehavioralEvent, User, UserSegment
from app.models.ml_models import BehavioralEventData
from app.database import get_db
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)

class BehavioralEventService:
    """Service for processing and analyzing behavioral events"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.event_batch = []
        self.batch_size = 10
        self.privacy_filters = [
            'password', 'token', 'secret', 'key', 'auth',
            'credit_card', 'ssn', 'social_security'
        ]

    async def track_event(
        self, 
        user_id: str, 
        event_type: str, 
        event_data: Dict[str, Any],
        db: Session
    ) -> bool:
        """Track a behavioral event asynchronously"""
        try:
            # Validate and sanitize event data
            sanitized_data = self._sanitize_event_data(event_data)
            
            # Create behavioral event data model
            behavioral_event_data = BehavioralEventData(
                event_type=event_type,
                **sanitized_data
            )

            # Process event immediately for critical events
            if self._is_critical_event(event_type):
                await self._process_event_immediately(user_id, behavioral_event_data, db)
            else:
                # Add to batch for non-critical events
                self.event_batch.append({
                    'user_id': user_id,
                    'event_data': behavioral_event_data,
                    'timestamp': datetime.utcnow()
                })

                # Process batch if it reaches batch size
                if len(self.event_batch) >= self.batch_size:
                    await self._process_event_batch(db)

            return True

        except Exception as e:
            logger.error(f"Error tracking event: {e}")
            return False

    async def track_page_view(
        self, 
        user_id: str, 
        page_url: str, 
        session_id: Optional[str] = None,
        db: Session = None
    ) -> bool:
        """Track page view event"""
        event_data = {
            'page_url': page_url,
            'session_id': session_id,
            'interaction_type': 'page_view'
        }
        return await self.track_event(user_id, 'page_view', event_data, db)

    async def track_recommendation_interaction(
        self,
        user_id: str,
        recommendation_id: str,
        interaction_type: str,  # 'view', 'click', 'dismiss', 'implement'
        session_id: Optional[str] = None,
        db: Session = None
    ) -> bool:
        """Track recommendation interaction event"""
        event_data = {
            'recommendation_id': recommendation_id,
            'interaction_type': interaction_type,
            'session_id': session_id
        }
        return await self.track_event(user_id, 'recommendation_interaction', event_data, db)

    async def track_purchase_behavior(
        self,
        user_id: str,
        product_id: str,
        action: str,  # 'view', 'add_to_cart', 'purchase', 'remove_from_cart'
        additional_data: Optional[Dict] = None,
        db: Session = None
    ) -> bool:
        """Track purchase-related behavior"""
        event_data = {
            'product_id': product_id,
            'interaction_type': action,
            'additional_data': additional_data or {}
        }
        return await self.track_event(user_id, 'purchase', event_data, db)

    async def track_goal_modification(
        self,
        user_id: str,
        goal_id: str,
        modification_type: str,  # 'create', 'update', 'delete', 'achieve'
        goal_data: Optional[Dict] = None,
        db: Session = None
    ) -> bool:
        """Track goal modification events"""
        event_data = {
            'goal_id': goal_id,
            'interaction_type': modification_type,
            'additional_data': goal_data or {}
        }
        return await self.track_event(user_id, 'goal_modification', event_data, db)

    async def process_event(self, user_id: str, event_data: BehavioralEventData, db: Session) -> bool:
        """Process individual behavioral event"""
        try:
            # Store event in database
            await self._store_event(user_id, event_data, db)

            # Analyze behavior change if needed
            if self._should_analyze_behavior_change(event_data.event_type):
                await self._analyze_behavior_change(user_id, db)

            return True

        except Exception as e:
            logger.error(f"Error processing event: {e}")
            return False

    async def batch_process_events(self, db: Session) -> int:
        """Process events in batch for efficiency"""
        try:
            if not self.event_batch:
                return 0

            processed_count = 0
            batch_to_process = self.event_batch.copy()
            self.event_batch.clear()

            # Process events in parallel
            tasks = []
            for event in batch_to_process:
                task = self._process_event_immediately(
                    event['user_id'], 
                    event['event_data'], 
                    db
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if not isinstance(result, Exception):
                    processed_count += 1
                else:
                    logger.error(f"Error in batch processing: {result}")

            logger.info(f"Batch processed {processed_count} events")
            return processed_count

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return 0

    async def aggregate_user_behavior(self, user_id: str, days: int = 30, db: Session = None) -> Dict:
        """Aggregate user behavior over specified period"""
        try:
            user_uuid = uuid.UUID(user_id)
            start_date = datetime.utcnow() - timedelta(days=days)

            # Query behavioral events
            events = db.query(BehavioralEvent)\
                .filter(
                    BehavioralEvent.user_id == user_uuid,
                    BehavioralEvent.timestamp >= start_date
                )\
                .all()

            # Aggregate by event type
            event_counts = {}
            interaction_patterns = {}
            
            for event in events:
                event_type = event.event_type
                event_counts[event_type] = event_counts.get(event_type, 0) + 1

                # Parse event data for interaction patterns
                try:
                    event_data = json.loads(event.event_data) if event.event_data else {}
                    interaction_type = event_data.get('interaction_type', 'unknown')
                    
                    if event_type not in interaction_patterns:
                        interaction_patterns[event_type] = {}
                    
                    interaction_patterns[event_type][interaction_type] = \
                        interaction_patterns[event_type].get(interaction_type, 0) + 1

                except json.JSONDecodeError:
                    continue

            # Calculate behavior metrics
            total_events = sum(event_counts.values())
            behavior_diversity = len(event_counts)
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(event_counts, total_events)

            return {
                'user_id': user_id,
                'period_days': days,
                'total_events': total_events,
                'event_counts': event_counts,
                'interaction_patterns': interaction_patterns,
                'behavior_diversity': behavior_diversity,
                'engagement_score': engagement_score,
                'generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error aggregating user behavior: {e}")
            return {}

    async def detect_behavior_changes(self, user_id: str, db: Session) -> List[Dict]:
        """Detect significant behavioral changes"""
        try:
            # Compare recent behavior (last 7 days) with historical (previous 30 days)
            recent_behavior = await self.aggregate_user_behavior(user_id, 7, db)
            historical_behavior = await self.aggregate_user_behavior(user_id, 30, db)

            changes = []

            if not recent_behavior or not historical_behavior:
                return changes

            # Detect changes in event frequency
            recent_counts = recent_behavior.get('event_counts', {})
            historical_counts = historical_behavior.get('event_counts', {})

            for event_type in set(list(recent_counts.keys()) + list(historical_counts.keys())):
                recent_count = recent_counts.get(event_type, 0)
                historical_avg = historical_counts.get(event_type, 0) / 4  # Weekly average

                if historical_avg > 0:
                    change_ratio = recent_count / historical_avg
                    
                    if change_ratio > 2.0:  # 100% increase
                        changes.append({
                            'type': 'increased_activity',
                            'event_type': event_type,
                            'change_ratio': change_ratio,
                            'description': f"Significant increase in {event_type} activity"
                        })
                    elif change_ratio < 0.5:  # 50% decrease
                        changes.append({
                            'type': 'decreased_activity',
                            'event_type': event_type,
                            'change_ratio': change_ratio,
                            'description': f"Significant decrease in {event_type} activity"
                        })

            # Detect new behavior patterns
            recent_patterns = set(recent_behavior.get('interaction_patterns', {}).keys())
            historical_patterns = set(historical_behavior.get('interaction_patterns', {}).keys())
            
            new_patterns = recent_patterns - historical_patterns
            for pattern in new_patterns:
                changes.append({
                    'type': 'new_behavior_pattern',
                    'pattern': pattern,
                    'description': f"New behavior pattern detected: {pattern}"
                })

            return changes

        except Exception as e:
            logger.error(f"Error detecting behavior changes: {e}")
            return []

    def _sanitize_event_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize event data to remove sensitive information"""
        sanitized = {}
        
        for key, value in event_data.items():
            # Skip sensitive fields
            if any(filter_term in key.lower() for filter_term in self.privacy_filters):
                continue
            
            # Sanitize nested dictionaries
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_event_data(value)
            elif isinstance(value, str) and len(value) > 1000:
                # Truncate very long strings
                sanitized[key] = value[:1000] + "..."
            else:
                sanitized[key] = value

        return sanitized

    def _is_critical_event(self, event_type: str) -> bool:
        """Determine if an event should be processed immediately"""
        critical_events = ['purchase', 'goal_modification', 'recommendation_interaction']
        return event_type in critical_events

    async def _process_event_immediately(
        self, 
        user_id: str, 
        event_data: BehavioralEventData, 
        db: Session
    ):
        """Process event immediately"""
        await self._store_event(user_id, event_data, db)
        
        # Trigger behavior analysis for critical events
        if self._should_analyze_behavior_change(event_data.event_type):
            await self._analyze_behavior_change(user_id, db)

    async def _process_event_batch(self, db: Session):
        """Process batch of events"""
        if not self.event_batch:
            return

        try:
            # Store all events in batch
            events_to_store = []
            for event in self.event_batch:
                user_uuid = uuid.UUID(event['user_id'])
                
                behavioral_event = BehavioralEvent(
                    user_id=user_uuid,
                    event_type=event['event_data'].event_type,
                    event_data=json.dumps(event['event_data'].dict()),
                    timestamp=event['timestamp'],
                    created_at=datetime.utcnow()
                )
                events_to_store.append(behavioral_event)

            # Bulk insert
            db.add_all(events_to_store)
            db.commit()

            logger.info(f"Batch stored {len(events_to_store)} events")
            
            # Clear batch
            self.event_batch.clear()

        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
            db.rollback()

    async def _store_event(self, user_id: str, event_data: BehavioralEventData, db: Session):
        """Store individual event in database"""
        try:
            user_uuid = uuid.UUID(user_id)
            
            behavioral_event = BehavioralEvent(
                user_id=user_uuid,
                event_type=event_data.event_type,
                event_data=json.dumps(event_data.dict()),
                timestamp=datetime.utcnow(),
                created_at=datetime.utcnow()
            )
            
            db.add(behavioral_event)
            db.commit()

        except Exception as e:
            logger.error(f"Error storing event: {e}")
            db.rollback()
            raise

    def _should_analyze_behavior_change(self, event_type: str) -> bool:
        """Determine if behavior change analysis should be triggered"""
        trigger_events = ['purchase', 'goal_modification', 'recommendation_interaction']
        return event_type in trigger_events

    async def _analyze_behavior_change(self, user_id: str, db: Session):
        """Analyze behavior change and update user segment if needed"""
        try:
            # Import here to avoid circular imports
            from app.services.ml_service import UserSegmentationService
            
            # Check if significant behavior change occurred
            changes = await self.detect_behavior_changes(user_id, db)
            
            if changes:
                # Update user segment if significant changes detected
                segmentation_service = UserSegmentationService()
                updated_segment = segmentation_service.update_user_segment(user_id, db)
                
                logger.info(f"Updated user segment for {user_id}: {updated_segment.segment_name}")

        except Exception as e:
            logger.error(f"Error analyzing behavior change: {e}")

    def _calculate_engagement_score(self, event_counts: Dict, total_events: int) -> float:
        """Calculate user engagement score based on event patterns"""
        if total_events == 0:
            return 0.0

        # Weight different event types
        event_weights = {
            'page_view': 1.0,
            'recommendation_interaction': 3.0,
            'purchase': 5.0,
            'goal_modification': 4.0
        }

        weighted_score = 0.0
        for event_type, count in event_counts.items():
            weight = event_weights.get(event_type, 1.0)
            weighted_score += count * weight

        # Normalize to 0-1 scale
        max_possible_score = total_events * max(event_weights.values())
        engagement_score = min(1.0, weighted_score / max_possible_score) if max_possible_score > 0 else 0.0

        return round(engagement_score, 3)

    async def get_user_behavior_summary(self, user_id: str, db: Session) -> Dict:
        """Get comprehensive user behavior summary"""
        try:
            # Get behavior aggregation
            behavior_data = await self.aggregate_user_behavior(user_id, 30, db)
            
            # Get recent behavior changes
            behavior_changes = await self.detect_behavior_changes(user_id, db)
            
            # Get user segment information
            user_uuid = uuid.UUID(user_id)
            user_segment = db.query(UserSegment)\
                .filter(UserSegment.user_id == user_uuid)\
                .first()

            segment_info = None
            if user_segment:
                segment_info = {
                    'segment_id': user_segment.segment_id,
                    'segment_name': user_segment.segment_name,
                    'confidence': float(user_segment.confidence),
                    'updated_at': user_segment.updated_at.isoformat()
                }

            return {
                'user_id': user_id,
                'behavior_data': behavior_data,
                'behavior_changes': behavior_changes,
                'segment_info': segment_info,
                'summary_generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting user behavior summary: {e}")
            return {}

    def validate_service_health(self) -> Dict:
        """Validate behavioral event service health"""
        return {
            'status': 'healthy',
            'batch_size': self.batch_size,
            'current_batch_count': len(self.event_batch),
            'executor_active': not self.executor._shutdown,
            'privacy_filters_count': len(self.privacy_filters),
            'last_check': datetime.utcnow().isoformat()
        }
