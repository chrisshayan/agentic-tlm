#!/usr/bin/env python3
"""
Simple test script for the TLM system.

This script tests basic functionality without external dependencies.
"""

import asyncio
import sys
import traceback

# Add src to path
sys.path.insert(0, 'src')


async def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.config.settings import settings
        print("✅ Settings imported successfully")
        
        from src.core.message_bus import MessageBus, Message, MessageType
        print("✅ Message bus imported successfully")
        
        from src.core.orchestrator import AgentOrchestrator
        print("✅ Orchestrator imported successfully")
        
        from src.agents.base_agent import BaseAgent
        print("✅ Base agent imported successfully")
        
        from src.agents.cffa import CashFlowForecastingAgent
        print("✅ CFFA imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


async def test_message_bus():
    """Test message bus functionality."""
    print("\nTesting message bus...")
    
    try:
        from src.core.message_bus import MessageBus, Message, MessageType
        message_bus = MessageBus()
        await message_bus.start()
        
        # Test message creation
        message = Message(
            message_type=MessageType.AGENT_HEARTBEAT,
            sender_id="test",
            payload={"test": "data"}
        )
        
        # Test publishing
        await message_bus.publish(message)
        
        await message_bus.stop()
        print("✅ Message bus test passed")
        return True
        
    except Exception as e:
        print(f"❌ Message bus test failed: {e}")
        traceback.print_exc()
        return False


async def test_agent_creation():
    """Test agent creation."""
    print("\nTesting agent creation...")
    
    try:
        from src.agents.cffa import CashFlowForecastingAgent
        
        agent = CashFlowForecastingAgent()
        print(f"✅ Created agent: {agent.agent_name}")
        
        status = agent.get_status()
        print(f"✅ Agent status: {status['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        traceback.print_exc()
        return False


async def test_orchestrator():
    """Test orchestrator functionality."""
    print("\nTesting orchestrator...")
    
    try:
        from src.core.orchestrator import AgentOrchestrator
        from src.core.message_bus import MessageBus
        from src.agents.cffa import CashFlowForecastingAgent
        
        message_bus = MessageBus()
        orchestrator = AgentOrchestrator(message_bus)
        
        # Create and register an agent
        agent = CashFlowForecastingAgent(message_bus)
        orchestrator.register_agent(agent)
        
        # Get status
        status = orchestrator.get_agent_status()
        print(f"✅ Orchestrator created with {status['agent_count']} agents")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        traceback.print_exc()
        return False


async def run_tests():
    """Run all tests."""
    print("=" * 50)
    print("TLM System - Basic Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_message_bus,
        test_agent_creation,
        test_orchestrator
    ]
    
    results = []
    
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to run.")
        print("\nTo start the full system, run:")
        print("  python run_tlm_system.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False
    
    return True


if __name__ == "__main__":
    print("Running basic TLM system tests...")
    
    try:
        result = asyncio.run(run_tests())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Test runner crashed: {e}")
        traceback.print_exc()
        sys.exit(1) 