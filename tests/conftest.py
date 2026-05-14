"""Test fixtures for openUBMC Code RAG tests."""

import pytest
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / "test_parsers" / "fixtures"


@pytest.fixture
def lua_sample():
    """Sample openUBMC Lua component code."""
    return '''local class = require 'mc.class'
local singleton = require 'mc.singleton'
local log = require 'mc.logging'

local SensorApp = class(base_app)

function SensorApp:init(config)
    self.config = config
    self.sensors = {}
    log:info("SensorApp initialized")
end

function SensorApp:pre_init()
    self._db = self:get_db()
end

function SensorApp:get_sensor_data(sensor_id)
    local data = self._db:select(self._db.Sensor)
        :where({id = sensor_id})
        :first()
    return data
end

function SensorApp:update_sensor(sensor_id, value)
    self._db:update(self._db.Sensor)
        :set({reading = value})
        :where({id = sensor_id})
        :execute()
    self:_notify_subscribers(sensor_id, value)
end

function SensorApp:_notify_subscribers(sensor_id, value)
    local subscribers = self._subscribers[sensor_id] or {}
    for _, callback in ipairs(subscribers) do
        callback(sensor_id, value)
    end
end

return SensorApp
'''


@pytest.fixture
def mds_service_json():
    """Sample MDS service.json."""
    return '''{
    "name": "sensor",
    "version": "1.0.0",
    "dependencies": [
        {"name": "mdb_interface", "version": ">=1.0.0"},
        {"name": "libipmi", "version": ">=1.0.0"}
    ],
    "required": [
        "bmc.kepler.Sensors",
        "bmc.kepler.SensorEvent"
    ]
}'''


@pytest.fixture
def mds_model_json():
    """Sample MDS model.json."""
    return '''{
    "ThresholdSensor": {
        "path": "/redfish/v1/Chassis/{ChassisId}/Sensors/{SensorId}",
        "properties": {
            "Reading": {"type": "number", "readonly": true},
            "UpperThresholdNonCritical": {"type": "number"},
            "UpperThresholdCritical": {"type": "number"},
            "LowerThresholdNonCritical": {"type": "number"},
            "SensorType": {"type": "string", "readonly": true}
        }
    },
    "DiscreteSensor": {
        "path": "/redfish/v1/Chassis/{ChassisId}/Sensors/{SensorId}",
        "properties": {
            "Reading": {"type": "integer", "readonly": true},
            "States": {"type": "array", "items": {"type": "string"}},
            "SensorType": {"type": "string", "readonly": true}
        }
    }
}'''


@pytest.fixture
def mds_ipmi_json():
    """Sample MDS ipmi.json."""
    return '''{
    "cmds": {
        "GetSensorReading": {
            "netfn": "0x04",
            "cmd": "0x2D",
            "request": {"sensor_number": {"type": "uint8"}},
            "response": {"reading": {"type": "uint8"}, "state": {"type": "uint8"}}
        },
        "SetSensorThresholds": {
            "netfn": "0x04",
            "cmd": "0x26",
            "request": {"sensor_number": {"type": "uint8"}, "thresholds": {"type": "bytes"}},
            "response": {"completion_code": {"type": "uint8"}}
        }
    }
}'''


@pytest.fixture
def c_sample():
    """Sample C code for libipmi."""
    return '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SENSOR_COUNT 256

typedef struct ipmi_sensor {
    uint8_t sensor_number;
    uint8_t owner_id;
    uint8_t sensor_type;
    char name[64];
    float reading;
} ipmi_sensor_t;

int ipmi_get_sensor_reading(uint8_t sensor_num, ipmi_sensor_t *sensor) {
    if (sensor == NULL) {
        return -1;
    }
    sensor->sensor_number = sensor_num;
    sensor->reading = 0.0f;
    return 0;
}

int ipmi_set_sensor_threshold(uint8_t sensor_num, float upper, float lower) {
    if (upper <= lower) {
        return -1;
    }
    return 0;
}

LUAMOD_API int luaopen_ipmi_sensor(lua_State *L) {
    luaL_Reg libs[] = {
        {"get_reading", l_get_sensor_reading},
        {"set_threshold", l_set_sensor_threshold},
        {NULL, NULL}
    };
    luaL_newlib(L, libs);
    return 1;
}
'''
